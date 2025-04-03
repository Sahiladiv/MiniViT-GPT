import os
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from .gpt2_prefix_caption_model import GPT2PrefixCaptionModel

# 🧠 Function to generate caption
def generate_caption(model, image_feat, context_embed, tokenizer, max_length=64):
    model.eval()
    with torch.no_grad():
        image_prefix = model.img_proj(image_feat).unsqueeze(1)
        context_prefix = model.ctx_proj(context_embed).unsqueeze(1)
        prefix = torch.cat([image_prefix, context_prefix], dim=1)

        generated_ids = []
        input_embeds = prefix
        attention_mask = torch.ones(1, 2).to(image_feat.device)

        for _ in range(max_length):
            output = model.gpt2(inputs_embeds=input_embeds, attention_mask=attention_mask)
            logits = output.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)

            generated_ids.append(next_token_id.item())

            next_token_embed = model.gpt2.transformer.wte(next_token_id)
            input_embeds = torch.cat([input_embeds, next_token_embed], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(attention_mask.device)], dim=1)

        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# 🏋️ Training function with validation
def training_model(dataset, val_dataset, tokenizer, save_dir="./checkpoints", num_epochs=10, batch_size=8, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2PrefixCaptionModel().to(device)

    # Train and validation loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps
    )

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(
                image_feat=batch['image_feat'],
                context_embed=batch['context_embed'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids']
            )

            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

        # Save model
        checkpoint_path = os.path.join(save_dir, f"gpt2_caption_epoch{epoch+1:02}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

        # === Validation ===
        model.eval()
        generated = []
        references = []

        for batch in tqdm(val_loader, desc=f"🔍 Validating (Epoch {epoch+1})"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            gen_caption = generate_caption(model, batch['image_feat'], batch['context_embed'], tokenizer)
            img_name = batch['filename'][0]
            gt_captions = dataset.caption_dict[img_name]  # get all 5 ground truths

            generated.append(gen_caption)
            references.append([ref.split() for ref in gt_captions])

        # === Evaluation Metrics ===
        smooth = SmoothingFunction().method1
        bleu1 = np.mean([sentence_bleu(ref, hyp.split(), weights=(1,0,0,0), smoothing_function=smooth) for hyp, ref in zip(generated, references)])
        bleu4 = np.mean([sentence_bleu(ref, hyp.split(), weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth) for hyp, ref in zip(generated, references)])

        # Format for CIDEr
        gts = {i: [" ".join(ref) for ref in refs] for i, refs in enumerate(references)}
        res = {i: [generated[i]] for i in range(len(generated))}
        cider_score, _ = Cider().compute_score(gts, res)

        print(f"\n📊 Validation (Epoch {epoch+1}):")
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")
        print(f"CIDEr:  {cider_score:.4f}\n")

    print("✅ Training & validation complete.")
