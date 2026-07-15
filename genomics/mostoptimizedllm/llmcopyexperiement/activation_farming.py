import os
import gc
import torch
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "Write a Python script to implement a double linked list.",
    "Explain the difference between classical and quantum mechanics.",
    "Prove that the square root of 2 is irrational.",
    "Compose a beautiful poem about space exploration.",
    "How does the immune system protect the human body?",
    "Solve the differential equation dy/dx = y*tan(x).",
    "Translate the following English passage into French and German: 'The quick brown fox jumps over the lazy dog.'",
    "What are the core principles of database normalization?",
    "Write a short story about an AI that learns to paint.",
    "Explain the concept of neural network gradient descent.",
    "Explain the theory of relativity in simple terms.",
    "How do you optimize SQL queries for large datasets?",
    "Write a marketing pitch for a new smart thermostat.",
    "Compare Hobbes and Locke's views on the social contract.",
    "How does a blockchain achieve consensus?",
    "Write a detailed recipe for making sourdough bread from scratch.",
    "Discuss the impact of the industrial revolution on society.",
    "Describe the life cycle of a star.",
    "What is the P vs NP problem in computer science?",
    "Detail the history of the silk road.",
]

def harvest_activations(model_id, target_tokens, save_dir, device="cuda"):
    print(f"Loading tokenizer and model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load model in bfloat16 for L4 GPU compatibility and low memory footprint
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Model has {num_layers} layers. Hidden size: {hidden_size}")

    os.makedirs(save_dir, exist_ok=True)

    # We will hook all MLP blocks and save inputs/outputs layer-by-layer.
    # To keep memory usage low, we will accumulate a list of tensors for each layer.
    farmed_inputs = {i: [] for i in range(num_layers)}
    farmed_outputs = {i: [] for i in range(num_layers)}
    tokens_collected = 0

    # Hook callback generator
    def get_hook(layer_idx):
        def hook(module, input_tensor, output_tensor):
            nonlocal tokens_collected
            # input_tensor is a tuple, first element is the input hidden state x
            x = input_tensor[0].detach().cpu() # Move to CPU to save GPU memory
            y = output_tensor.detach().cpu()
            
            # Reshape from [batch, seq_len, hidden] to [-1, hidden]
            x_flat = x.view(-1, hidden_size)
            y_flat = y.view(-1, hidden_size)
            
            farmed_inputs[layer_idx].append(x_flat)
            farmed_outputs[layer_idx].append(y_flat)
        return hook

    # Register hooks
    hooks = []
    for i in range(num_layers):
        mlp_block = model.model.layers[i].mlp
        h = mlp_block.register_forward_hook(get_hook(i))
        hooks.append(h)

    print("Starting generation loop for activation farming...")
    pbar = tqdm(total=target_tokens, desc="Tokens Farmed")
    
    random.shuffle(PROMPTS)
    prompt_idx = 0

    with torch.no_grad():
        while tokens_collected < target_tokens:
            prompt = PROMPTS[prompt_idx % len(PROMPTS)]
            # Add some randomness to prompts to increase diversity
            if prompt_idx >= len(PROMPTS):
                prompt += f" Answer style variation {prompt_idx}."
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]
            
            # Generate synthetic sequence with high temperature to explore the manifold
            generation_outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            
            new_tokens = generation_outputs.shape[1] - prompt_len
            tokens_collected += prompt_len + new_tokens
            pbar.update(prompt_len + new_tokens)
            prompt_idx += 1
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

    # Remove hooks
    for h in hooks:
        h.remove()

    print("\nFarming completed. Consolidating and saving tensors to disk...")
    # For each layer, concatenate all collected tensors and save to disk
    for i in range(num_layers):
        x_layer = torch.cat(farmed_inputs[i], dim=0)[:target_tokens]
        y_layer = torch.cat(farmed_outputs[i], dim=0)[:target_tokens]
        
        x_path = os.path.join(save_dir, f"x_layer_{i}.pt")
        y_path = os.path.join(save_dir, f"y_layer_{i}.pt")
        
        torch.save(x_layer, x_path)
        torch.save(y_layer, y_path)
        print(f"Layer {i}: Saved {x_layer.shape[0]} tokens to {x_path}")
        
        # Free memory immediately
        farmed_inputs[i] = None
        farmed_outputs[i] = None
        gc.collect()

    print(f"All activation tensors saved successfully under {save_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase A: Activation Farming for Gemma-3 EML-KAN translation")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="Model to farm activations from")
    parser.add_argument("--target_tokens", type=int, default=100000, help="Number of token activation vectors to collect")
    parser.add_argument("--save_dir", type=str, default="farmed_activations", help="Directory to save activation tensors")
    args = parser.parse_args()
    
    # Run activation farming
    harvest_activations(args.model_id, args.target_tokens, args.save_dir)
