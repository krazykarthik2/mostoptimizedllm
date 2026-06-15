import torch
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.float32)
emb = model.model.text_model.get_input_embeddings().weight
print(f"Embedding Mean: {emb.mean().item():.6f}")
print(f"Embedding Std: {emb.std().item():.6f}")
