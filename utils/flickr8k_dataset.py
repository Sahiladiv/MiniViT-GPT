from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, ViTModel, ViTFeatureExtractor


DATASET_PATH = 'Flickr8K'
filepath = f'{DATASET_PATH}/captions.txt'

VIT_MODEL_PATH = "google/vit-base-patch16-224-in21k"

# Load ViT model + extractor once
vit_model = ViTModel.from_pretrained(VIT_MODEL_PATH).eval().cuda()
vit_extractor = ViTFeatureExtractor.from_pretrained(VIT_MODEL_PATH)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class Flickr8kDataset(Dataset):
    def __init__(self, csv_path, image_dir, gpt2_max_len=64):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.minilm = minilm_model
        self.max_len = gpt2_max_len

        # Load CSV
        df = pd.read_csv(csv_path)
        self.caption_dict = {}
        for img, cap in zip(df['image'], df['caption']):
            self.caption_dict.setdefault(img, []).append(cap)

        # Keep only images with 5 captions
        self.data = [(img, caps) for img, caps in self.caption_dict.items() if len(caps) == 5]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, captions = self.data[idx]

        # === 1. Load raw image using PIL ===
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # === 2. Extract ViT CLS feature ===
        inputs = vit_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            vit_out = vit_model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        image_feat = vit_out.squeeze(0).cpu()  # shape: [768]

        # === 3. Pick one caption as target, others as context ===
        target_idx = np.random.randint(0, 5)
        target_caption = captions[target_idx]
        context_captions = [cap for i, cap in enumerate(captions) if i != target_idx]
        context_text = " ".join(context_captions)

        # === 4. MiniLM context embedding ===
        context_embed = self.minilm.encode(context_text, convert_to_tensor=True).float()  # shape: [384]

        # === 5. Tokenize target caption ===
        tokenized = self.tokenizer(
            target_caption,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'image_feat': image_feat,  # [768]
            'context_embed': context_embed,  # [384]
            'input_ids': tokenized['input_ids'].squeeze(0),  # [max_len]
            'attention_mask': tokenized['attention_mask'].squeeze(0),  # [max_len]
            'filename': img_name
        }