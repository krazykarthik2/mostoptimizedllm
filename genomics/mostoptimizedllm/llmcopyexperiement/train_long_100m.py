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

def generate_synthetic_data_long(num_samples=25000):
    names = ["John", "Alice", "Bob", "Emma", "David", "Mary", "James", "Sarah", "Emily", "Michael", "Sophia", "Daniel"]
    items = ["apples", "oranges", "books", "pens", "coins", "candies", "marbles", "stamps", "notebooks", "rulers"]
    
    examples = []
    
    # 1. Math Reasoning (Multi-Step Logic)
    for _ in range(num_samples):
        n1 = random.choice(names)
        n2 = random.choice(names)
        while n2 == n1:
            n2 = random.choice(names)
        item = random.choice(items)
        start = random.randint(5, 50)
        give = random.randint(1, start - 2)
        buy = random.randint(2, start)
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
        
    # 2. Python Code Generation
    coding_tasks = [
        ("reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("check if a year is leap", "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)"),
        ("find the maximum number in a list", "def find_max(lst):\n    return max(lst) if lst else None"),
        ("calculate factorial", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
        ("check if a number is even", "def is_even(n):\n    return n % 2 == 0"),
        ("calculate fibonacci up to n terms", "def fib(n):\n    a, b = 0, 1\n    res = []\n    for _ in range(n):\n        res.append(a)\n        a, b = b, a + b\n    return res"),
        ("convert celsius to fahrenheit", "def c_to_f(c):\n    return c * 9/5 + 32"),
        ("check if a string is palindrome", "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]")
    ]
    for _ in range(num_samples):
        desc, code = random.choice(coding_tasks)
        prompt = f"Write a python function to {desc}."
        answer = f"Here is the Python implementation:\n\n```python\n{code}\n```"
        examples.append((prompt, answer))
        
    # 3. JSON Agent Tool-Calling trajectories
    tool_tasks = [
        ("Calculate 8374 * 2839 using the calculator tool.", "calculator", "8374 * 2839", "23773786", "23,773,786"),
        ("What is the capital of France? Search the web.", "google_search", "capital of France", "Paris is the capital of France.", "Paris"),
        ("Compute (847 + 293) * 12 using the calculator.", "calculator", "(847 + 293) * 12", "13680", "13,680"),
        ("Search Wikipedia for python programming language.", "wikipedia", "python programming language", "Python is an interpreted, high-level, general-purpose programming language.", "Python"),
        ("Multiply 298.5 by 49 using the math tool.", "calculator", "298.5 * 49", "14626.5", "14,626.5")
    ]
    for _ in range(num_samples):
        prompt, tool, arg, response, result = random.choice(tool_tasks)
        answer = (
            f"[TOOL_CALL] {{\n  \"tool\": \"{tool}\",\n  \"args\": {{\"{'expression' if tool=='calculator' else 'query'}\": \"{arg}\"}}\n}}\n"
            f"[TOOL_RESPONSE] {response}\n"
            f"Based on the tool output, the result is {result}."
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

class StreamingMassiveDataset(Dataset):
    def __init__(self, tokenizer, seq_len=256, max_blocks=50000):
        self.examples = []
        all_tokens = []
        
        # 1. Compile cached wikitext
        print("Packing local Wikitext-2-raw corpus...")
        raw_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for row in raw_wikitext:
            text = row["text"].strip()
            if text:
                all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
                
        # 2. Compile massive synthetic reasoning/coding/tools datasets
        print("Generating and packing 105,000 synthetic logic/code/tool samples...")
        synthetic_examples = generate_synthetic_data_long(num_samples=35000)
        for prompt, answer in synthetic_examples:
            messages = [{"role": "user", "content": prompt}, {"role": "model", "content": answer}]
            sanitized = sanitize_messages(messages)
            formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
            all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
            
        # Chunk into sequences
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
            if len(self.examples) >= max_blocks:
                break
                
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
    
    # Compile the 100M token dataset blocks
    train_dataset = StreamingMassiveDataset(tokenizer, seq_len=256, max_blocks=50000)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
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
            
    from muon import MuonWithAuxAdam
    # Base learning rate: 1e-5 (for Muon), adam_lr: 2e-5 (for norms)
    base_lr = 1e-5
    optimizer = MuonWithAuxAdam(model, lr=base_lr, adam_lr=2e-5)
    
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
