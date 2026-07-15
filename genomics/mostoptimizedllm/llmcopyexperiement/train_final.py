import os
import time
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANGatedMLP, Gemma3HopfieldKANAttention
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ==============================================================================
# 1. Dataset Generation (Math Reasoning, Code Gen, and JSON Tool Calls)
# ==============================================================================

def generate_synthetic_data(num_samples=1000):
    names = ["John", "Alice", "Bob", "Emma", "David", "Mary", "James", "Sarah", "Emily", "Michael"]
    items = ["apples", "oranges", "books", "pens", "coins", "candies", "marbles"]
    
    examples = []
    
    # 1. Math Reasoning (Multi-Step Logic)
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
        
    # 2. Python Code Generation
    coding_tasks = [
        ("reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("check if a year is leap", "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)"),
        ("find the maximum number in a list", "def find_max(lst):\n    return max(lst) if lst else None"),
        ("calculate factorial", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
        ("check if a number is even", "def is_even(n):\n    return n % 2 == 0")
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
        ("Compute (847 + 293) * 12 using the calculator.", "calculator", "(847 + 293) * 12", "13680", "13,680")
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
# 2. Dataset Packing
# ==============================================================================

class BalancedMasterDataset(Dataset):
    def __init__(self, raw_guanaco, synthetic_examples, tokenizer, seq_len=256, max_blocks=4000):
        self.examples = []
        all_tokens = []
        
        # 1. Format synthetic examples
        print("Packing synthetic math, coding, and tool datasets...")
        for prompt, answer in synthetic_examples:
            messages = [{"role": "user", "content": prompt}, {"role": "model", "content": answer}]
            sanitized = sanitize_messages(messages)
            formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
            all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
            
        # 2. Format Guanaco conversational data (representing grammatical structure & formatting)
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
                
        # Chunk into blocks
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
# 3. Evaluation and Training Control
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

def run_master_training():
    model_id = "google/gemma-3-1b-it"
    save_tuned_path = "gemma3_eml_kan/model_state_master_final.pt"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Generating synthetic datasets...")
    synthetic_examples = generate_synthetic_data(num_samples=1000)
    
    print("Loading raw openassistant-guanaco dataset...")
    raw_guanaco = load_dataset("timdettmers/openassistant-guanaco", split="train")

    train_dataset = BalancedMasterDataset(raw_guanaco, synthetic_examples, tokenizer, seq_len=256, max_blocks=4000)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda:0")
    
    print("Swapping self_attn to Gemma3HopfieldKANAttention and FFN to Gemma3EMLKANGatedMLP...")
    for i in range(model.config.num_hidden_layers):
        # Swap Attention
        orig_attn = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = Gemma3HopfieldKANAttention(orig_attn).to("cuda:0")
        
        # Swap MLP and copy weights 1:1
        orig_mlp = model.model.layers[i].mlp
        kan_mlp = Gemma3EMLKANGatedMLP(model.config).to(torch.bfloat16).to("cuda:0")
        with torch.no_grad():
            kan_mlp.gate_proj.linear.weight.copy_(orig_mlp.gate_proj.weight)
            kan_mlp.up_proj.weight.copy_(orig_mlp.up_proj.weight)
            kan_mlp.down_proj.weight.copy_(orig_mlp.down_proj.weight)
        model.model.layers[i].mlp = kan_mlp
        
    # Unfreeze only the EML-KAN MLP blocks and Normalization layers (Attention remains frozen)
    print("Unfreezing EML-KAN MLP and Normalization layers (Attention remains frozen)...")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "mlp" in name or "norm" in name or "ln_" in name:
            param.requires_grad = True
            
    # Optimize with AdamW at stable lr = 1e-5
    base_lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    test_prompts = [
        "Hello! I am John and I have 5 apples. If I give 2 apples to Mary and buy 3 more apples from the store, how many apples do I have now? Explain your reasoning step-by-step.",
        "Write a python function to reverse a string.",
        "Search the web for the capital of France and tell me."
    ]
    
    model.train()
    step_count = 0
    max_steps = 2000
    
    print(f"Starting Final Master Calibration training for {max_steps} steps...")
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
                print(f"\n[Early Stopping Triggered] Loss reached optimal threshold: {loss.item():.4f} at step {step_count}.")
                step_count = max_steps
                break
                
            if step_count % 1000 == 0:
                model.eval()
                print(f"\n[Step {step_count}] Running generation checks...")
                for p in test_prompts:
                    response = run_evaluation(model, tokenizer, p)
                    print(f"Prompt: {p}\nResponse:\n{response}\n")
                model.train()
                
    total_time = time.time() - t_start
    print(f"Completed calibration in {total_time:.2f} seconds.")
    
    # Save the calibrated weights
    print(f"Saving final master calibrated weights to {save_tuned_path}...")
    torch.save(model.state_dict(), save_tuned_path)
    
    # Final eval
    model.eval()
    print("\n" + "="*80)
    print("          FINAL EVALUATION OF FULL KAN-HYBRID LLM MODEL")
    print("="*80)
    for p in test_prompts:
        response = run_evaluation(model, tokenizer, p)
        print(f"Prompt: {p}\nResponse:\n{response}\n")
    print("="*80)

if __name__ == "__main__":
    run_master_training()
