import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, ViTModel, ViTFeatureExtractor



class GPT2PrefixCaptionModel(nn.Module):
    def __init__(self, gpt2_model_name='gpt2', image_dim=768, context_dim=384, prefix_len=2):
        super().__init__()
        
        self.prefix_len = prefix_len

        # Load GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.gpt2_embedding_dim = self.gpt2.config.n_embd  # usually 768 for gpt2

        # Projection layers
        self.img_proj = nn.Linear(image_dim, self.gpt2_embedding_dim)
        self.ctx_proj = nn.Linear(context_dim, self.gpt2_embedding_dim)

    def forward(self, image_feat, context_embed, input_ids, attention_mask, labels=None):
        B = image_feat.size(0)

        # === 1. Project prefix ===
        image_prefix = self.img_proj(image_feat).unsqueeze(1)        # [B, 1, 768]
        context_prefix = self.ctx_proj(context_embed).unsqueeze(1)   # [B, 1, 768]
        prefix = torch.cat([image_prefix, context_prefix], dim=1)    # [B, 2, 768]

        # === 2. Token embeddings from input_ids ===
        input_embeds = self.gpt2.transformer.wte(input_ids)          # [B, seq_len, 768]

        # === 3. Concatenate full input ===
        full_input = torch.cat([prefix, input_embeds], dim=1)        # [B, prefix + seq_len, 768]

        # === 4. Adjust attention mask ===
        prefix_mask = torch.ones(B, self.prefix_len).to(attention_mask.device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, prefix + seq_len]

        # === 5. Adjust labels to ignore prefix ===
        if labels is not None:
            # Pad labels with -100 to ignore loss on prefix tokens
            ignore_prefix = torch.full((B, self.prefix_len), -100).to(labels.device)
            full_labels = torch.cat([ignore_prefix, labels], dim=1)  # [B, prefix + seq_len]
        else:
            full_labels = None

        # === 6. Forward pass ===
        outputs = self.gpt2(
            inputs_embeds=full_input,
            attention_mask=full_attention_mask,
            labels=full_labels
        )

        return outputs