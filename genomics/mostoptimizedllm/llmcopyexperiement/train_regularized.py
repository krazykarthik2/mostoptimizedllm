import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from muon import MuonWithAuxAdam

# ==============================================================================
# 1. Dataset Compilation
# ==============================================================================

def generate_synthetic_data_reg(num_samples=5000):
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
        ("check if a number is even", "def is_even(n):\n    return n % 2 == 0")
    ]
    for _ in range(num_samples):
        desc, code = random.choice(coding_tasks)
        prompt = f"Write a python function to {desc}."
        answer = f"Here is the Python implementation:\n\n```python\n{code}\n```"
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

class RegularizedDataset(Dataset):
    def __init__(self, tokenizer, seq_len=256, max_blocks=3000):
        self.examples = []
        all_tokens = []
        
        # 1. Wikitext-2 for grammar
        print("Packing Wikitext-2 corpus...")
        raw_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for row in raw_wikitext:
            text = row["text"].strip()
            if text:
                all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
                
        # 2. Synthetic samples
        print("Generating and packing synthetic samples...")
        synthetic_examples = generate_synthetic_data_reg(num_samples=3000)
        for prompt, answer in synthetic_examples:
            messages = [{"role": "user", "content": prompt}, {"role": "model", "content": answer}]
            sanitized = sanitize_messages(messages)
            formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
            all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
            
        # Chunk
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
            if len(self.examples) >= max_blocks:
                break
                
        print(f"Dataset compiled: {len(self.examples)} blocks.")
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

# ==============================================================================
# 2. Training Loop with KL-Divergence Distillation
# ==============================================================================

def run_evaluation(model, tokenizer, prompt_str):
    messages = [{"role": "user", "content": prompt_str}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    model_id = "google/gemma-3-1b-it"
    save_path = "gemma3_eml_kan/model_state_regularized.pt"
    device = "cuda:0"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    train_dataset = RegularizedDataset(tokenizer, seq_len=256, max_blocks=3000)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # 1. Load Teacher Model (Frozen, unmodified)
    print("Loading frozen teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    
    # 2. Load Student Model (EML-KAN Swapped)
    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    
    # Swap layers
    for i in range(student.config.num_hidden_layers):
        orig_mlp = student.model.layers[i].mlp
        kan_mlp = Gemma3EMLKANGatedMLP(student.config).to(torch.bfloat16).to(device)
        with torch.no_grad():
            kan_mlp.gate_proj.linear.weight.copy_(orig_mlp.gate_proj.weight)
            kan_mlp.up_proj.weight.copy_(orig_mlp.up_proj.weight)
            kan_mlp.down_proj.weight.copy_(orig_mlp.down_proj.weight)
        student.model.layers[i].mlp = kan_mlp
        
    # Unfreeze only the EML-KAN MLP layers and Normalization layers
    student.requires_grad_(False)
    for name, param in student.named_parameters():
        if "mlp" in name or "norm" in name or "ln_" in name:
            param.requires_grad = True
            
    # Optimizer with strong weight decay (0.05) to penalize non-linear coefficients
    base_lr = 1e-5
    optimizer = MuonWithAuxAdam(student, lr=base_lr, adam_lr=2e-5, weight_decay=0.05)
    
    test_prompts = [
        "If a train travels 60 miles per hour, how far will it travel in 2.5 hours? Explain your reasoning step-by-step.",
        "A father has 4 daughters. Each daughter has a brother. How many children does the father have in total? Explain your reasoning.",
        "Write a python function to find the largest element in a list of numbers."
    ]
    
    student.train()
    step_count = 0
    max_steps = 1500
    temperature = 2.0
    alpha = 0.7  # 70% KL loss weight, 30% CE loss weight
    
    print(f"Starting Knowledge Distillation training for {max_steps} steps...")
    t_start = time.time()
    
    while step_count < max_steps:
        for batch in loader:
            if step_count >= max_steps:
                break
                
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * step_count / max_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad()
            inputs = batch.to(device)
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()
            
            # Forward pass teacher
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                teacher_logits = teacher_outputs.logits.detach()
                
            # Forward pass student
            student_outputs = student(inputs)
            student_logits = student_outputs.logits
            
            # 1. Hard cross-entropy loss
            loss_ce = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)).float(),
                targets.view(-1)
            )
            
            # 2. Soft KL-Divergence loss
            p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
            loss_kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)
            
            # Combine losses
            loss = (1 - alpha) * loss_ce + alpha * loss_kl
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}/{max_steps} | Loss: {loss.item():.4f} | KL: {loss_kl.item():.4f} | CE: {loss_ce.item():.4f}")
                
            if step_count % 500 == 0:
                student.eval()
                print(f"\n[Step {step_count}] Running generation check...")
                for p in test_prompts:
                    response = run_evaluation(student, tokenizer, p)
                    print(f"Prompt: {p}\nResponse:\n{response}\n")
                student.train()
                
    # Save the calibrated weights
    print(f"Saving regularized weights to {save_path}...")
    torch.save(student.state_dict(), save_path)
    
    # Final eval
    student.eval()
    print("\n" + "="*80)
    print("          FINAL EVALUATION OF REGULARIZED KAN-HYBRID LLM MODEL")
    print("="*80)
    for p in test_prompts:
        response = run_evaluation(student, tokenizer, p)
        print(f"Prompt: {p}\nResponse:\n{response}\n")
    print("="*80)

if __name__ == "__main__":
    main()
