import os
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Gemma3EMLKANMLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# 1. Synthesize structured reasoning examples
def generate_synthetic_reasoning():
    names = ["John", "Alice", "Bob", "Emma", "David", "Mary", "James", "Sarah"]
    items = ["apples", "oranges", "books", "pens", "coins", "candies", "marbles", "peaches"]
    
    examples = []
    
    # Template: basic transaction
    for i in range(100):
        name1 = random.choice(names)
        name2 = random.choice(names)
        while name2 == name1:
            name2 = random.choice(names)
        item = random.choice(items)
        
        start = random.randint(5, 20)
        give = random.randint(1, 4)
        buy = random.randint(2, 8)
        final = start - give + buy
        
        prompt = (
            f"Hello! I am {name1} and I have {start} {item}. If I give {give} {item} to {name2} "
            f"and buy {buy} more {item} from the store, how many {item} do I have now? Explain your reasoning step-by-step."
        )
        
        answer = (
            f"Let's break it down step-by-step:\n\n"
            f"1. **Start:** You begin with {start} {item}.\n"
            f"2. **Give to {name2}:** You give {give} {item} to {name2}, so you have {start} - {give} = {start - give} {item}.\n"
            f"3. **Buy more:** You buy {buy} more {item}, so you now have {start - give} + {buy} = {final} {item}.\n\n"
            f"**Therefore, you now have {final} {item}.**"
        )
        examples.append((prompt, answer))
        
    return examples

class ReasoningDataset(Dataset):
    def __init__(self, raw_dataset, synthetic_examples, tokenizer, seq_len=384, max_blocks=1000):
        self.examples = []
        all_tokens = []
        
        # Format and append synthetic math/reasoning examples
        print("Formatting synthetic reasoning dataset...")
        for prompt, answer in synthetic_examples:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "model", "content": answer}
            ]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            
        # Blend in conversational data from Guanaco
        print("Packing openassistant-guanaco reasoning sequences...")
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
            
            # Focus on conversational turns that contain reasoning triggers
            if len(messages) > 0:
                has_reasoning = any(word in messages[0]["content"].lower() for word in ["explain", "why", "reason", "step", "calculate", "solve", "math", "logical"])
                if has_reasoning or random.random() < 0.3:
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
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_reasoning_calibration():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state.pt"
    save_tuned_path = "gemma3_eml_kan/model_state_reasoning.pt"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    synthetic_examples = generate_synthetic_reasoning()
    
    print("Loading openassistant-guanaco dataset...")
    raw_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

    train_dataset = ReasoningDataset(raw_dataset, synthetic_examples, tokenizer, seq_len=384, max_blocks=1000)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16
    ).to("cuda:0")
    
    print("Swapping MLP blocks to PURE Gemma3EMLKANMLP...")
    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].mlp = Gemma3EMLKANMLP(model.config).to(torch.bfloat16).to("cuda:0")
        
    print(f"Loading fitted weights from {weights_path}...")
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
    
    test_prompt = (
        "Hello! I am John and I have 5 apples. If I give 2 apples to Mary and buy 3 "
        "more apples from the store, how many apples do I have now? Explain your reasoning step-by-step."
    )
    
    model.train()
    step_count = 0
    max_steps = 3000
    
    print(f"Starting reasoning calibration training for {max_steps} steps...")
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
                
            if step_count % 1000 == 0:
                model.eval()
                print(f"\n[Step {step_count}] Running generation check...")
                response = run_evaluation(model, tokenizer, test_prompt)
                print(f"Response:\n{response}\n")
                model.train()
                
    total_time = time.time() - t_start
    print(f"Completed {max_steps} steps in {total_time:.2f} seconds.")
    
    # Save the calibrated state dict
    print(f"Saving reasoning calibrated weights to {save_tuned_path}...")
    torch.save(model.state_dict(), save_tuned_path)
    
    # Final eval
    model.eval()
    print("\n--- FINAL EVALUATION OF PURE REASONING ALIGNED EML-KAN MODEL (GPU) ---")
    response = run_evaluation(model, tokenizer, test_prompt)
    print(f"Response:\n{response}")

if __name__ == "__main__":
    run_reasoning_calibration()
