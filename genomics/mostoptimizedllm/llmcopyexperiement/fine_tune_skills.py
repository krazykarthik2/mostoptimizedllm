import os
import time
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANMLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# 1. Generate Synthetic Coding & Tool Calling Examples
def generate_skills_dataset():
    examples = []
    
    # --- Coding Examples ---
    coding_tasks = [
        ("Write a python function to check if a number is prime.", 
         "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ("Write a python function to reverse a string.",
         "def reverse_string(s):\n    return s[::-1]"),
        ("Write a python function to calculate the factorial of a number.",
         "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("Write a python function to find the maximum number in a list.",
         "def find_max(lst):\n    if not lst:\n        return None\n    max_val = lst[0]\n    for val in lst:\n        if val > max_val:\n            max_val = val\n    return max_val"),
        ("Write a python function to check if a string is a palindrome.",
         "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]")
    ]
    # Multiply and randomize variations to make 50 examples
    for i in range(50):
        prompt, code = random.choice(coding_tasks)
        answer = f"Here is the Python implementation:\n\n```python\n{code}\n```"
        examples.append((prompt, answer))
        
    # --- Tool Calling Examples ---
    # We define a calculator tool and a google search tool
    tool_scenarios = [
        ("Calculate 8374 * 2839 using the calculator tool.", 
         "[TOOL_CALL] {\"tool\": \"calculator\", \"args\": {\"expression\": \"8374 * 2839\"}}\n[TOOL_RESPONSE] 23773786\nBased on the calculator tool, 8374 multiplied by 2839 is 23,773,786."),
        ("What is the capital of France? Search the web.",
         "[TOOL_CALL] {\"tool\": \"google_search\", \"args\": {\"query\": \"capital of France\"}}\n[TOOL_RESPONSE] Paris is the capital and most populous city of France.\nAccording to search results, the capital of France is Paris."),
        ("Find the population of Tokyo. Search the web.",
         "[TOOL_CALL] {\"tool\": \"google_search\", \"args\": {\"query\": \"population of Tokyo\"}}\n[TOOL_RESPONSE] Tokyo's population is approximately 14 million people.\nThe web search indicates that the population of Tokyo is approximately 14 million."),
        ("Compute (847 + 293) * 12 using the calculator.",
         "[TOOL_CALL] {\"tool\": \"calculator\", \"args\": {\"expression\": \"(847 + 293) * 12\"}}\n[TOOL_RESPONSE] 13680\nThe result of the calculation is 13,680."),
        ("Search the web for the height of Mount Everest.",
         "[TOOL_CALL] {\"tool\": \"google_search\", \"args\": {\"query\": \"height of Mount Everest\"}}\n[TOOL_RESPONSE] Mount Everest is 8848.86 meters tall.\nWeb search shows that Mount Everest is 8,848.86 meters tall.")
    ]
    for i in range(50):
        prompt, answer = random.choice(tool_scenarios)
        examples.append((prompt, answer))
        
    # --- Math Reasoning Examples ---
    math_examples = []
    names = ["John", "Alice", "Bob", "Emma", "David"]
    items = ["apples", "oranges", "books", "pens"]
    for i in range(50):
        name1 = random.choice(names)
        name2 = random.choice(names)
        while name2 == name1:
            name2 = random.choice(names)
        item = random.choice(items)
        start = random.randint(5, 15)
        give = random.randint(1, 4)
        buy = random.randint(2, 6)
        final = start - give + buy
        prompt = f"Hello! I am {name1} and I have {start} {item}. If I give {give} {item} to {name2} and buy {buy} more {item}, how many do I have now? Explain step-by-step."
        answer = f"Let's break it down:\n1. Start: You have {start} {item}.\n2. Give: You have {start} - {give} = {start-give} {item}.\n3. Buy: You now have {start-give} + {buy} = {final} {item}.\nTherefore, you have {final} {item} now."
        math_examples.append((prompt, answer))
        
    examples.extend(math_examples)
    return examples

class SkillsDataset(Dataset):
    def __init__(self, raw_dataset, skills_examples, tokenizer, seq_len=256, max_blocks=1000):
        self.examples = []
        all_tokens = []
        
        # Format and append synthetic skills examples
        print("Formatting synthetic skills dataset...")
        for prompt, answer in skills_examples:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "model", "content": answer}
            ]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            
        # Blend in conversational data from Guanaco
        print("Packing openassistant-guanaco sequences...")
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
                # Prioritize coding and tool calling queries
                has_trigger = any(word in messages[0]["content"].lower() for word in ["code", "python", "function", "write a program", "calculator", "search", "web", "find"])
                if has_trigger or random.random() < 0.2:
                    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                    tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                    all_tokens.extend(tokens)
                    
            if len(all_tokens) >= seq_len * max_blocks:
                break
                
        # Chunk into fixed-length blocks
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
            if len(self.examples) >= max_blocks:
                break
        print(f"Dataset compiled: {len(self.examples)} packed blocks of length {seq_len}.")
                    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

def run_evaluation(model, tokenizer, prompt_str):
    messages = [{"role": "user", "content": prompt_str}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_skills_calibration():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state.pt"
    save_tuned_path = "gemma3_eml_kan/model_state_skills.pt"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    skills_examples = generate_skills_dataset()
    
    print("Loading openassistant-guanaco dataset...")
    raw_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

    train_dataset = SkillsDataset(raw_dataset, skills_examples, tokenizer, seq_len=256, max_blocks=800)
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
        "Write a python function to reverse a string.",
        "Search the web for the capital of France and tell me."
    ]
    
    model.train()
    step_count = 0
    max_steps = 3000
    
    print(f"Starting skills calibration training for {max_steps} steps...")
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
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}/{max_steps} | Loss: {loss.item():.4f}")
                
            if step_count % 500 == 0:
                model.eval()
                print(f"\n[Step {step_count}] Running generation checks...")
                for p in test_prompts:
                    response = run_evaluation(model, tokenizer, p)
                    print(f"Prompt: {p}\nResponse:\n{response}\n")
                model.train()
                
    total_time = time.time() - t_start
    print(f"Completed {max_steps} steps in {total_time:.2f} seconds.")
    
    # Save the calibrated state dict
    print(f"Saving skills calibrated weights to {save_tuned_path}...")
    torch.save(model.state_dict(), save_tuned_path)
    
    # Final eval
    model.eval()
    print("\n--- FINAL EVALUATION OF PURE SKILLS ALIGNED EML-KAN MODEL (GPU) ---")
    for p in test_prompts:
        response = run_evaluation(model, tokenizer, p)
        print(f"Prompt: {p}\nResponse:\n{response}\n")

if __name__ == "__main__":
    run_skills_calibration()
