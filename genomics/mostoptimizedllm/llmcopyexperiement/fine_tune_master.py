import os
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANMLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ==============================================================================
# 1. High-Volume Synthetic Dataset Generation
# ==============================================================================

def generate_large_synthetic_data(num_samples=1000):
    names = ["John", "Alice", "Bob", "Emma", "David", "Mary", "James", "Sarah", "Emily", "Michael"]
    items = ["apples", "oranges", "books", "pens", "coins", "candies", "marbles", "peaches"]
    
    examples = []
    
    # 1. Math Reasoning (1000 samples)
    for _ in range(num_samples):
        n1 = random.choice(names)
        n2 = random.choice(names)
        while n2 == n1:
            n2 = random.choice(names)
        item = random.choice(items)
        start = random.randint(5, 25)
        give = random.randint(1, 4)
        buy = random.randint(2, 10)
        final = start - give + buy
        prompt = f"Hello! I am {n1} and I have {start} {item}. If I give {give} {item} to {n2} and buy {buy} more {item}, how many do I have now? Explain step-by-step."
        answer = (
            f"Let's break it down step-by-step:\n"
            f"1. Start: You have {start} {item}.\n"
            f"2. Give: You give {give} to {n2}, leaving you with {start} - {give} = {start - give} {item}.\n"
            f"3. Buy: You buy {buy} more, so you have {start - give} + {buy} = {final} {item}.\n"
            f"Therefore, you have {final} {item} now."
        )
        examples.append((prompt, answer))
        
    # 2. Python Coding Tasks (1000 samples)
    coding_functions = [
        ("check if a number is prime", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"),
        ("reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("calculate factorial", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
        ("find the maximum number in a list", "def find_max(lst):\n    return max(lst) if lst else None"),
        ("check if a string is a palindrome", "def is_palindrome(s):\n    c = ''.join(x.lower() for x in s if x.isalnum())\n    return c == c[::-1]"),
        ("check if a year is leap", "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)"),
        ("merge two sorted lists", "def merge_lists(l1, l2):\n    return sorted(l1 + l2)")
    ]
    for _ in range(num_samples):
        desc, code = random.choice(coding_functions)
        prompt = f"Write a python function to {desc}."
        answer = f"Here is the Python implementation:\n\n```python\n{code}\n```"
        examples.append((prompt, answer))
        
    # 3. Tool Calling Scenarios (1000 samples)
    tool_tasks = [
        ("Calculate 8374 * 2839 using the calculator tool.", "calculator", "8374 * 2839", "23773786", "23,773,786"),
        ("What is the capital of France? Search the web.", "google_search", "capital of France", "Paris is the capital of France.", "Paris"),
        ("Find the population of Tokyo. Search the web.", "google_search", "population of Tokyo", "Tokyo has 14 million people.", "14 million"),
        ("Compute (847 + 293) * 12 using the calculator.", "calculator", "(847 + 293) * 12", "13680", "13,680"),
        ("Search the web for the height of Mount Everest.", "google_search", "height of Mount Everest", "Mount Everest is 8848.86m tall.", "8,848.86 meters")
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

# ==============================================================================
# 2. Packed Master Dataset Builder
# ==============================================================================

class MasterDataset(Dataset):
    def __init__(self, raw_guanaco, raw_wikitext, synthetic_examples, tokenizer, seq_len=256, max_blocks=10000):
        self.examples = []
        all_tokens = []
        
        # 1. Format synthetic examples
        print("Packing synthetic dataset...")
        for prompt, answer in synthetic_examples:
            messages = [{"role": "user", "content": prompt}, {"role": "model", "content": answer}]
            sanitized = sanitize_messages(messages)
            formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
            all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
            
        # 2. Format Guanaco dataset (all 9846 rows)
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
                
        # 3. Format Wikitext dataset
        print("Packing wikitext Wikipedia corpus...")
        for row in raw_wikitext:
            text = row["text"].strip()
            if text:
                all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
                
        # Chunk the massive stream into fixed-length blocks
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
            if len(self.examples) >= max_blocks:
                break
                
        print(f"Master Dataset compiled: {len(self.examples)} packed blocks of length {seq_len}.")
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

# ==============================================================================
# 3. Main Calibration Training Run
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

def run_master_calibration():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state.pt"
    save_tuned_path = "gemma3_eml_kan/model_state_master.pt"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Generating large-scale synthetic datasets...")
    synthetic_examples = generate_large_synthetic_data(num_samples=1000)
    
    print("Loading raw openassistant-guanaco dataset...")
    raw_guanaco = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    print("Loading raw wikitext dataset...")
    raw_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    train_dataset = MasterDataset(raw_guanaco, raw_wikitext, synthetic_examples, tokenizer, seq_len=256, max_blocks=12000)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16
    ).to("cuda:0")
    
    print("Swapping MLP blocks to PURE Gemma3EMLKANMLP...")
    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].mlp = Gemma3EMLKANMLP(model.config).to(torch.bfloat16).to("cuda:0")
        
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cuda:0")
    model.load_state_dict(state_dict)
    
    # Unfreeze KAN MLP blocks AND all Norm layers
    print("Unfreezing EML-KAN blocks and all Normalization layers...")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "mlp" in name or "norm" in name or "ln_" in name:
            param.requires_grad = True
            
    # Optimize with AdamW at lr = 4e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    
    test_prompts = [
        "Hello! I am John and I have 5 apples. If I give 2 apples to Mary and buy 3 more apples from the store, how many apples do I have now? Explain your reasoning step-by-step.",
        "Write a python function to reverse a string.",
        "Search the web for the capital of France and tell me."
    ]
    
    model.train()
    step_count = 0
    max_steps = 5000
    
    print(f"Starting Master Calibration training for {max_steps} steps...")
    t_start = time.time()
    
    while step_count < max_steps:
        for batch in loader:
            if step_count >= max_steps:
                break
                
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
            
            if step_count % 200 == 0:
                print(f"  Step {step_count}/{max_steps} | Loss: {loss.item():.4f}")
                
            if step_count % 1000 == 0:
                model.eval()
                print(f"\n[Step {step_count}] Running generation checks...")
                for p in test_prompts:
                    response = run_evaluation(model, tokenizer, p)
                    print(f"Prompt: {p}\nResponse:\n{response}\n")
                model.train()
                
    total_time = time.time() - t_start
    print(f"Completed {max_steps} steps in {total_time:.2f} seconds.")
    
    # Save the calibrated state dict
    print(f"Saving master calibrated weights to {save_tuned_path}...")
    torch.save(model.state_dict(), save_tuned_path)
    
    # Final eval
    model.eval()
    print("\n" + "="*80)
    print("          FINAL EVALUATION OF PURE MASTER ALIGNED EML-KAN MODEL")
    print("="*80)
    for p in test_prompts:
        response = run_evaluation(model, tokenizer, p)
        print(f"Prompt: {p}\nResponse:\n{response}\n")
    print("="*80)

if __name__ == "__main__":
    run_master_calibration()
