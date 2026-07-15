import os
import time
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ==============================================================================
# 1. Massive Data Streaming & Packing (100M Tokens Target)
# ==============================================================================

class StreamingMassiveDataset(Dataset):
    def __init__(self, tokenizer, seq_len=256, max_tokens=100000000):
        self.examples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print("Streaming massive datasets from HF Hub...")
        
        # Load datasets
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        gsm8k = load_dataset("gsm8k", "main", split="train", streaming=True)
        code = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        
        all_tokens = []
        
        # Stream & mix tokens
        wiki_iter = iter(wiki)
        gsm_iter = iter(gsm8k)
        code_iter = iter(code)
        
        total_tokens_collected = 0
        print("Collecting 100M tokens...")
        
        while total_tokens_collected < max_tokens:
            try:
                # 1. Grammar & General Knowledge (Wiki)
                row = next(wiki_iter)
                text = row["text"].strip()
                if text:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(tokens)
                    total_tokens_collected += len(tokens)
            except StopIteration:
                pass
                
            try:
                # 2. Logic & Math (GSM8K)
                row = next(gsm_iter)
                prompt = row["question"]
                answer = row["answer"]
                messages = [{"role": "user", "content": prompt}, {"role": "model", "content": answer}]
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                all_tokens.extend(tokens)
                total_tokens_collected += len(tokens)
            except StopIteration:
                pass
                
            try:
                # 3. Python Code Generation (CodeAlpaca)
                row = next(code_iter)
                prompt = row["instruction"]
                output = row["output"]
                messages = [{"role": "user", "content": prompt}, {"role": "model", "content": output}]
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                all_tokens.extend(tokens)
                total_tokens_collected += len(tokens)
            except StopIteration:
                pass
                
            # If all iterators are exhausted, break
            if not wiki_iter and not gsm_iter and not code_iter:
                break
                
            # Keep the list size bounded in memory by chunking periodically
            if len(all_tokens) >= seq_len * 5000:
                for i in range(0, len(all_tokens) - seq_len, seq_len):
                    self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
                all_tokens = []
                
        # Handle residual tokens
        if len(all_tokens) >= seq_len:
            for i in range(0, len(all_tokens) - seq_len, seq_len):
                self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
                
        print(f"Dataset compiled: {len(self.examples)} packed blocks of length {seq_len} (approx. {len(self.examples)*seq_len/1e6:.2f}M tokens).")
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

# ==============================================================================
# 2. Main Long Training Pipeline
# ==============================================================================

def main():
    model_id = "google/gemma-3-1b-it"
    checkpoint_dir = "gemma3_eml_kan/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = "cuda:0"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Pack up to 100M tokens
    train_dataset = StreamingMassiveDataset(tokenizer, seq_len=256, max_tokens=100000000)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True) # Batch size 4, gradient accumulation steps 2 = effective batch size 8
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    
    print("Swapping MLP blocks to Gemma3EMLKANGatedMLP (Attention kept native)...")
    for i in range(model.config.num_hidden_layers):
        orig_mlp = model.model.layers[i].mlp
        kan_mlp = Gemma3EMLKANGatedMLP(model.config).to(torch.bfloat16).to(device)
        with torch.no_grad():
            kan_mlp.gate_proj.linear.weight.copy_(orig_mlp.gate_proj.weight)
            kan_mlp.up_proj.weight.copy_(orig_mlp.up_proj.weight)
            kan_mlp.down_proj.weight.copy_(orig_mlp.down_proj.weight)
        model.model.layers[i].mlp = kan_mlp
        
    # Unfreeze only the EML-KAN MLP layers and Normalization layers
    print("Unfreezing EML-KAN MLP and Normalization layers...")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "mlp" in name or "norm" in name or "ln_" in name:
            param.requires_grad = True
            
    # Base learning rate: 1e-5
    base_lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    model.train()
    step_count = 0
    max_steps = 50000
    grad_accum_steps = 2
    
    print(f"Starting long training session for {max_steps} steps (equivalent to 100M tokens)...")
    t_start = time.time()
    
    optimizer.zero_grad()
    
    while step_count < max_steps:
        for batch in loader:
            if step_count >= max_steps:
                break
                
            # Cosine learning rate scheduler
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * step_count / max_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            inputs = batch.to(device)
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()
            
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)).float(),
                targets.view(-1)
            )
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (step_count + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            step_count += 1
            
            if step_count % 100 == 0:
                elapsed = time.time() - t_start
                tokens_processed = step_count * 4 * 256
                tps = tokens_processed / elapsed
                print(f"  Step {step_count}/{max_steps} | Loss: {loss.item() * grad_accum_steps:.4f} | LR: {lr:.2e} | Speed: {tps:.1f} tokens/s")
                
            # Save checkpoints periodically
            if step_count % 5000 == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{step_count}.pt"
                print(f"\n[Step {step_count}] Saving checkpoint to {checkpoint_path}...")
                torch.save(model.state_dict(), checkpoint_path)
                
    total_time = time.time() - t_start
    print(f"Completed long training in {total_time/3600:.2f} hours.")
    
    # Save the final model weights
    final_path = "gemma3_eml_kan/model_state_long_100m.pt"
    print(f"Saving final trained model to {final_path}...")
    torch.save(model.state_dict(), final_path)

if __name__ == "__main__":
    main()
