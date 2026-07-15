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
# 1. High-Density Mathematical & Logical Dataset Generation
# ==============================================================================

def generate_math_focused_data():
    names = ["John", "Alice", "Bob", "Emma", "David", "Mary", "James", "Sarah", "Emily", "Michael"]
    items = ["apples", "oranges", "books", "pens", "coins", "candies", "marbles", "peaches"]
    
    examples = []
    
    # 1. Simple Arithmetic Transactions (1200 samples)
    for _ in range(1200):
        n1 = random.choice(names)
        n2 = random.choice(names)
        while n2 == n1:
            n2 = random.choice(names)
        item = random.choice(items)
        
        start = random.randint(3, 15)
        give = random.randint(1, start - 1)
        buy = random.randint(2, 10)
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
        
    # 2. Probability & Marbles Tasks (800 samples)
    for _ in range(800):
        red = random.randint(3, 8)
        blue = random.randint(3, 8)
        take_red = random.randint(1, 2)
        
        new_red = red - take_red
        total = new_red + blue
        
        prompt = (
            f"If a box contains {red} red balls and {blue} blue balls, and I take out {take_red} red balls, "
            f"what is the probability of drawing a red ball next? Explain your reasoning step-by-step."
        )
        answer = (
            f"Let's break it down step-by-step:\n\n"
            f"1. **Analyze Initial State:** The box contains {red} red balls and {blue} blue balls (Total = {red + blue} balls).\n"
            f"2. **Remove Balls:** You take out {take_red} red balls, leaving {red} - {take_red} = {new_red} red balls.\n"
            f"3. **Count Remaining Total:** The remaining balls are {new_red} red and {blue} blue. The new total is {new_red} + {blue} = {total} balls.\n"
            f"4. **Calculate Probability:** The probability of drawing a red ball next is the number of remaining red balls divided by the new total.\n\n"
            f"**Therefore, the probability is {new_red}/{total}.**"
        )
        examples.append((prompt, answer))

    # 3. Python Coding Tasks (500 samples)
    coding_functions = [
        ("reverse a string", "def reverse_string(s):\n    return s[::-1]"),
        ("check if a year is leap", "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)"),
        ("find the maximum number in a list", "def find_max(lst):\n    return max(lst) if lst else None"),
        ("calculate the square of a number", "def square(n):\n    return n * n")
    ]
    for _ in range(500):
        desc, code = random.choice(coding_functions)
        prompt = f"Write a python function to {desc}."
        answer = f"Here is the Python implementation:\n\n```python\n{code}\n```"
        examples.append((prompt, answer))
        
    # 4. Tool Calling Scenarios (500 samples)
    tool_tasks = [
        ("Calculate 8374 * 2839 using the calculator tool.", "calculator", "8374 * 2839", "23773786", "23,773,786"),
        ("What is the capital of France? Search the web.", "google_search", "capital of France", "Paris is the capital of France.", "Paris"),
        ("Compute (847 + 293) * 12 using the calculator.", "calculator", "(847 + 293) * 12", "13680", "13,680")
    ]
    for _ in range(500):
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
# 2. Packed Dataset Builder (Math-Focused)
# ==============================================================================

class MathFocusedDataset(Dataset):
    def __init__(self, raw_guanaco, synthetic_examples, tokenizer, seq_len=256, max_blocks=5000):
        self.examples = []
        all_tokens = []
        
        # 1. Format synthetic examples
        print("Packing synthetic math/logic dataset...")
        for prompt, answer in synthetic_examples:
            messages = [{"role": "user", "content": prompt}, {"role": "model", "content": answer}]
            sanitized = sanitize_messages(messages)
            formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
            all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
            
        # 2. Format filtered Guanaco dataset
        print("Packing conversational math/coding sequences...")
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
                has_trigger = any(word in messages[0]["content"].lower() for word in ["calculate", "math", "logical", "reasoning", "explain", "code", "python", "solve"])
                if has_trigger or random.random() < 0.2:
                    sanitized = sanitize_messages(messages)
                    formatted_text = tokenizer.apply_chat_template(sanitized, tokenize=False)
                    all_tokens.extend(tokenizer.encode(formatted_text, add_special_tokens=False))
                    
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

def run_math_focused_calibration():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state.pt"
    save_tuned_path = "gemma3_eml_kan/model_state_math_focused.pt"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Generating math-focused synthetic datasets...")
    synthetic_examples = generate_math_focused_data()
    
    print("Loading raw openassistant-guanaco dataset...")
    raw_guanaco = load_dataset("timdettmers/openassistant-guanaco", split="train")

    train_dataset = MathFocusedDataset(raw_guanaco, synthetic_examples, tokenizer, seq_len=256, max_blocks=6000)
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
        "If a box contains 3 red balls and 5 blue balls, and I take out 2 red balls, what is the probability of drawing a red ball next? Explain your reasoning step-by-step."
    ]
    
    model.train()
    step_count = 0
    max_steps = 3000
    
    print(f"Starting Math-Focused Calibration training for {max_steps} steps...")
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
    
    # Save the calibrated weights
    print(f"Saving math-focused calibrated weights to {save_tuned_path}...")
    torch.save(model.state_dict(), save_tuned_path)
    
    # Final eval
    model.eval()
    print("\n" + "="*80)
    print("          FINAL EVALUATION OF PURE MATH-FOCUSED ALIGNED EML-KAN MODEL")
    print("="*80)
    for p in test_prompts:
        response = run_evaluation(model, tokenizer, p)
        print(f"Prompt: {p}\nResponse:\n{response}\n")
    print("="*80)

if __name__ == "__main__":
    run_math_focused_calibration()
