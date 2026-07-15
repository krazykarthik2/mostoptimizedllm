import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANMLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

class PackedGuanacoDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, seq_len=256, max_blocks=400):
        self.examples = []
        
        # Concatenate all tokens from formatted conversations
        all_tokens = []
        print("Packing conversational sequences...")
        for row in raw_dataset:
            text = row["text"]
            parts = text.split("###")
            messages = []
            for part in parts:
                part = part.strip()
                if part.startswith("Human:"):
                    messages.append({"role": "user", "content": part[6:].strip()})
                elif part.startswith("Assistant:"):
                    messages.append({"role": "model", "content": part[10:].strip()})
            
            if len(messages) > 0:
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                # Tokenize and append to stream
                tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                all_tokens.extend(tokens)
                
            if len(all_tokens) >= seq_len * max_blocks:
                break
                
        # Chunk into fixed-length blocks
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
            if len(self.examples) >= max_blocks:
                break
                
        print(f"Packed dataset contains {len(self.examples)} blocks of length {seq_len}.")
                    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

def run_packed_calibration():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state.pt"
    save_tuned_path = "gemma3_eml_kan/model_state_packed.pt"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading timdettmers/openassistant-guanaco dataset...")
    raw_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

    train_dataset = PackedGuanacoDataset(raw_dataset, tokenizer, seq_len=256, max_blocks=300)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16
    ).to("cuda:0")
    
    print("Swapping MLP blocks to Gemma3EMLKANMLP...")
    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].mlp = Gemma3EMLKANMLP(model.config).to(torch.bfloat16).to("cuda:0")
        
    print(f"Loading fitted weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cuda:0")
    model.load_state_dict(state_dict)
    
    # UNFREEZE KAN MLP BLOCKS (keep attention/norms frozen to preserve base features)
    print("Unfreezing EML-KAN MLP blocks for calibration...")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "mlp" in name:
            param.requires_grad = True
            
    # Optimize with AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    model.train()
    t0 = time.time()
    print("Starting joint packed Guanaco calibration training (2 epochs)...")
    for epoch in range(2):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            inputs = batch.to("cuda:0")
            
            # Autoregressive next-token prediction
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()
            
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)).float(),
                targets.view(-1)
            )
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if step % 25 == 0:
                print(f"  Epoch {epoch+1} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}")
        
    training_time = time.time() - t0
    print(f"Packed calibration completed in {training_time:.2f} seconds.")
    
    # Save the calibrated state dict
    print(f"Saving packed calibrated weights to {save_tuned_path}...")
    torch.save(model.state_dict(), save_tuned_path)
    
    # Run evaluation on OOD prompt
    model.eval()
    custom_prompt = (
        "Hello! I am John and I have 5 apples. If I give 2 apples to Mary and buy 3 "
        "more apples from the store, how many apples do I have now? Explain your reasoning step-by-step."
    )
    messages = [{"role": "user", "content": custom_prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\n--- EVALUATING PACKED ALIGNED EML-KAN MODEL (GPU) ---")
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Response:\n{response}")

if __name__ == "__main__":
    run_packed_calibration()
