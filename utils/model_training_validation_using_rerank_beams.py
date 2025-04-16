import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_scheduler, GPT2Tokenizer
from torch.optim import AdamW
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
import spacy
from .gpt2_prefix_caption_model import GPT2PrefixCaptionModel

nlp = spacy.load("en_core_web_sm")

# === Concept extractor ===
def extract_concepts(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "VERB"}]

# === Semantic scorer ===
def semantic_score(pred_concepts, context_concepts):
    return len(set(pred_concepts) & set(context_concepts))

# === Re-ranker ===
def rerank_beams(beams, context_text):
    context_concepts = extract_concepts(context_text)
    scored = []

    for beam in beams:
        beam_concepts = extract_concepts(beam)
        score = semantic_score(beam_concepts, context_concepts)
        scored.append((beam, score))

    return max(scored, key=lambda x: x[1])[0]

# === Greedy decoding ===
def generate_caption_greedy(model, image_feat, context_embed, tokenizer, max_length=64):
    model.eval()
    with torch.no_grad():
        image_prefix = model.img_proj(image_feat).unsqueeze(1)
        context_prefix = model.ctx_proj(context_embed).unsqueeze(1)
        prefix = torch.cat([image_prefix, context_prefix], dim=1)

        input_embeds = prefix
        attention_mask = torch.ones(1, 2).to(image_feat.device)
        generated_ids = []

        for _ in range(max_length):
            output = model.gpt2(inputs_embeds=input_embeds, attention_mask=attention_mask)
            logits = output.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)

            generated_ids.append(next_token_id.item())

            next_token_embed = model.gpt2.transformer.wte(next_token_id)
            input_embeds = torch.cat([input_embeds, next_token_embed], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(image_feat.device)], dim=1)

        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# === Beam search decoding ===
def generate_caption_beam(model, image_feat, context_embed, tokenizer, max_length=64, beam_width=3, return_top_k=False):
    model.eval()
    device = image_feat.device

    with torch.no_grad():
        image_prefix = model.img_proj(image_feat).unsqueeze(1)
        context_prefix = model.ctx_proj(context_embed).unsqueeze(1)
        prefix = torch.cat([image_prefix, context_prefix], dim=1)

        beams = [{
            "tokens": [],
            "logprob": 0.0,
            "input_embeds": prefix.clone(),
            "attention_mask": torch.ones(1, prefix.shape[1], device=device)
        }]

        for _ in range(max_length):
            all_candidates = []
            for beam in beams:
                output = model.gpt2(inputs_embeds=beam["input_embeds"], attention_mask=beam["attention_mask"])
                logits = output.logits[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    token_id = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    token_logprob = topk_log_probs[0, i].item()

                    token_embed = model.gpt2.transformer.wte(token_id)
                    new_input_embeds = torch.cat([beam["input_embeds"], token_embed], dim=1)
                    new_attention_mask = torch.cat([beam["attention_mask"], torch.ones(1, 1, device=device)], dim=1)

                    candidate = {
                        "tokens": beam["tokens"] + [token_id.item()],
                        "logprob": beam["logprob"] + token_logprob,
                        "input_embeds": new_input_embeds,
                        "attention_mask": new_attention_mask
                    }
                    all_candidates.append(candidate)

            beams = sorted(all_candidates, key=lambda x: x["logprob"], reverse=True)[:beam_width]

        if return_top_k:
            return [tokenizer.decode(beam["tokens"], skip_special_tokens=True).strip() for beam in beams]
        else:
            best = beams[0]["tokens"]
            return tokenizer.decode(best, skip_special_tokens=True).strip()

# === Training function ===
def training_model(dataset, val_dataset, tokenizer, save_dir="./checkpoints", num_epochs=10,
                   batch_size=8, lr=1e-4, decode_strategy="greedy", beam_width=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2PrefixCaptionModel().to(device)

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
        print(f"\n✅ Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(save_dir, f"gpt2_caption_epoch{epoch+1:02}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

        # === Validation ===
        model.eval()
        generated = []
        references = []

        for batch in tqdm(val_loader, desc=f"🔍 Validating (Epoch {epoch+1})"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            image_feat = batch['image_feat']
            context_embed = batch['context_embed']
            img_name = batch['filename'][0]

            if decode_strategy == "beam":
                top_beams = generate_caption_beam(model, image_feat, context_embed, tokenizer,
                                                  beam_width=beam_width, return_top_k=True)
                context_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
                gen_caption = rerank_beams(top_beams, context_text)
            else:
                gen_caption = generate_caption_greedy(model, image_feat, context_embed, tokenizer)

            gt_captions = dataset.caption_dict[img_name]
            generated.append(gen_caption)
            references.append([ref.split() for ref in gt_captions])

        smooth = SmoothingFunction().method1
        bleu1 = np.mean([sentence_bleu(ref, hyp.split(), weights=(1, 0, 0, 0), smoothing_function=smooth)
                         for hyp, ref in zip(generated, references)])
        bleu4 = np.mean([sentence_bleu(ref, hyp.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
                         for hyp, ref in zip(generated, references)])

        gts = {i: [" ".join(ref) for ref in refs] for i, refs in enumerate(references)}
        res = {i: [generated[i]] for i in range(len(generated))}
        cider_score, _ = Cider().compute_score(gts, res)

        print(f"\n📊 Validation (Epoch {epoch+1}):")
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")
        print(f"CIDEr:  {cider_score:.4f}\n")

    print("\n✅ Training & validation complete.")
