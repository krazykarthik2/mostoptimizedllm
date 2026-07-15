import os
import time
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANMLP, Gemma3HopfieldKANAttention
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ==============================================================================
# 1. Dataset Generation & Packing
# ==============================================================================

def generate_reasoning_data(num_samples=500):
    names = ["John", "Alice", "Bob", "Emma", "David", "Mary", "James", "Sarah"]
    items = ["apples", "oranges", "books", "pens", "coins", "candies", "marbles"]
    examples = []
    
    # Math Reasoning
    for _ in range(num_samples):
        n1 = random.choice(names)
        n2 = random.choice(names)
        while n2 == n1:
            n2 = random.choice(names)
        item = random.choice(items)
        start = random.randint(5, 20)
        give = random.randint(1, 4)
        buy = random.randint(2, 8)
        final = start - give + buy
        prompt = f"Hello! I am {n1} and I have {start} {item}. If I give {give} {item} to {n2} and buy {buy} more {item} from the store, how many {item} do I have now? Explain your reasoning step-by-step."
        answer = (
            f"Let's break it down step-by-step:\n\n"
            f"1. **Start:** You begin with {start} {item}.\n"
            f"2. **Give to {n2}:** You give {give} {item} to {n2}, so you have {start} - {give} = {start - give} {item}.\n"
            f"3. **Buy more:** You buy {buy} more {item}, so you now have {start - give} + {buy} = {final} {item}.\n\n"
            f"**Therefore, you now have {final} {item}.**"
        )
        examples.append((prompt, answer))
        
    return examples

def sanitize_messages(messages):
    sanitized = []
    current_role = None
    for msg in messages:
        if not msg["content"].strip():
            continue
        if msg["role"] != current_role:
            sanitized.append(msg)
            current_role = msg["role"]
        else:
            sanitized[-1]["content"] += "\n" + msg["content"]
    return sanitized

class StableCalibrationDataset(Dataset):
    def __init__(self, raw_guanaco, raw_wikitext, synthetic_examples, tokenizer, seq_len=256, max_blocks=5000):
        self.examples = []
        all_tokens = []
        
        # 1. Format synthetic examples
        print("Packing synthetic math dataset...")
        for prompt, answer in synthetic_examples:
            messages = [{"role": "user", "content": prompt}, {"role": "model", "content": answer}]
            sanitized = sanitize_messages(messages)
            formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
            all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
            
        # 2. Format Guanaco conversational data
        print("Packing openassistant-guanaco dataset...")
        for row in raw_guanaco:
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
                sanitized = sanitize_messages(messages)
                formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
                all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
                
        # 3. Format Wikitext dataset (excellent for perplexity and grammar regularization)
        print("Packing wikitext corpus...")
        for row in raw_wikitext:
            text = row["text"].strip()
            if text:
                all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
                
        # Chunk the massive stream into fixed-length blocks
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
            if len(self.examples) >= max_blocks:
                break
                
        print(f"Dataset compiled: {len(self.examples)} packed blocks of length {seq_len}.")
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

# ==============================================================================
# 2. Main Calibration Loop with Cosine Scheduler and Early Stopping
# ==============================================================================

def run_evaluation(model, tokenizer, prompt_str):
    messages = [{"role": "user", "content": prompt_str}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_stable_calibration():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state.pt"
    save_tuned_path = "gemma3_eml_kan/model_state_stable.pt"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Generating math-reasoning datasets...")
    synthetic_examples = generate_reasoning_data(num_samples=500)
    
    print("Loading raw openassistant-guanaco dataset...")
    raw_guanaco = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    print("Loading raw wikitext dataset...")
    raw_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    train_dataset = StableCalibrationDataset(raw_guanaco, raw_wikitext, synthetic_examples, tokenizer, seq_len=256, max_blocks=5000)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16
    ).to("cuda:0")
    
    print("Swapping self_attn to Gemma3HopfieldKANAttention and FFN to pure Gemma3EMLKANMLP...")
    for i in range(model.config.num_hidden_layers):
        # Swap MLP
        model.model.layers[i].mlp = Gemma3EMLKANMLP(model.config).to(torch.bfloat16).to("cuda:0")
        # Swap Attention
        orig_attn = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = Gemma3HopfieldKANAttention(orig_attn).to("cuda:0")
        
    print(f"Loading fitted weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cuda:0")
    
    hopfield_state_dict = {}
    for k, v in state_dict.items():
        if "self_attn." in k:
            new_key = k.replace("self_attn.", "self_attn.original_attn.")
            hopfield_state_dict[new_key] = v
        else:
            hopfield_state_dict[k] = v
            
    model.load_state_dict(hopfield_state_dict, strict=False)
    
    # Unfreeze only the EML-KAN MLP blocks and Normalization layers (Attention remains frozen)
    print("Unfreezing EML-KAN MLP and Normalization layers (Attention remains frozen)...")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "mlp" in name or "norm" in name or "ln_" in name:
            param.requires_grad = True
            
    # Stable learning rate: 1e-5
    base_lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    test_prompt = (
        "Hello! I am John and I have 5 apples. If I give 2 apples to Mary and buy 3 "
        "more apples from the store, how many apples do I have now? Explain your reasoning step-by-step."
    )
    
    model.train()
    step_count = 0
    max_steps = 1500
    
    print(f"Starting stable calibration training for {max_steps} steps...")
    t_start = time.time()
    
    while step_count < max_steps:
        for batch in loader:
            if step_count >= max_steps:
                break
                
            # Cosine learning rate scheduler
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * step_count / max_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad()
            inputs = batch.to("cuda:0")
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
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
                
            # Early stopping check: if loss drops below 2.5, stop training to prevent overfitting!
            if loss.item() < 2.5 and step_count > 400:
                print(f"\n[Early Stopping Triggered] Loss reached perplexity threshold: {loss.item():.4f} at step {step_count}.")
                step_count = max_steps
                break
                
            if step_count % 500 == 0:
                model.eval()
                print(f"\n[Step {step_count}] Running generation check...")
                response = run_evaluation(model, tokenizer, test_prompt)
                print(f"Response:\n{response}\n")
                model.train()
                
    total_time = time.time() - t_start
    print(f"Completed calibration in {total_time:.2f} seconds.")
    
    # Save the calibrated weights
    print(f"Saving stable weights to {save_tuned_path}...")
    torch.save(model.state_dict(), save_tuned_path)
    
    # Final eval
    model.eval()
    print("\n" + "="*80)
    print("          FINAL EVALUATION OF STABLE KAN-HYBRID LLM MODEL")
    print("="*80)
    response = run_evaluation(model, tokenizer, test_prompt)
    print(response)
    print("="*80)

if __name__ == "__main__":
    run_stable_calibration()
