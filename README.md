# MiniViT-GPT: Context-Aware Image Captioning

MiniViT-GPT is a multimodal Transformer-based framework that generates context-aware image captions and coherent short stories by integrating visual inputs with associated textual contexts. The system leverages Vision Transformers (ViT), lightweight text encoders, and GPT-2 language models using a parameter-efficient prefix tuning strategy.

### Project Overview
MiniViT-GPT is designed to bridge vision and language understanding by generating semantically aligned, context-enriched captions and narratives for images. The model conditions a frozen GPT-2 decoder with fused embeddings from image features (ViT) and textual context (MiniLM) using prefix tuning, enabling efficient adaptation without full model fine-tuning.

Image Input --> Vision Transformer (ViT-B/16) --> Image Embedding

Context Text --> MiniLM-L6-v2 --> Context Embedding

[Image + Context Embedding] --> Prefix Tuning Layer --> Injected into GPT-2 Decoder

GPT-2 Decoder --> Autoregressive Text Generation (Caption / Story)


