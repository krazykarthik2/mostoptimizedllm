import os
import torch
from model import Gemma3HopfieldKANAttention, Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate_model_weights(weights_name, weights_path, tokenizer, test_prompts, device):
    print("\n" + "="*80)
    print(f" TESTING: {weights_name} ({weights_path})")
    print("="*80)
    
    # Load model config and instantiate
    model_id = "google/gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    
    # Swap layers
    for i in range(model.config.num_hidden_layers):
        orig_attn = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = Gemma3HopfieldKANAttention(orig_attn).to(device)
        model.model.layers[i].mlp = Gemma3EMLKANGatedMLP(model.config).to(torch.bfloat16).to(device)
        
    # Load weights
    print(f"Loading weights state dict from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    for p in test_prompts:
        messages = [{"role": "user", "content": p}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\nPrompt: {p}\nResponse:\n{response}")
        print("-"*80)

def main():
    model_id = "google/gemma-3-1b-it"
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    test_prompts = [
        "If a train travels 60 miles per hour, how far will it travel in 2.5 hours? Explain your reasoning step-by-step.",
        "A father has 4 daughters. Each daughter has a brother. How many children does the father have in total? Explain your reasoning.",
        "Write a python function to find the largest element in a list of numbers.",
        "What is the largest ocean on Earth?"
    ]
    
    # 1. Test Calibrated Weights
    weights_path_final = "gemma3_eml_kan/model_state_master_final.pt"
    if os.path.exists(weights_path_final):
        evaluate_model_weights("Calibrated Master Model", weights_path_final, tokenizer, test_prompts, device)
    else:
        print(f"Calibrated weights not found at {weights_path_final}")
        
    # 2. Test Zero-Error Copy Weights
    weights_path_zero = "gemma3_eml_kan/model_state_zero_error.pt"
    if os.path.exists(weights_path_zero):
        evaluate_model_weights("Zero-Error Copy Model", weights_path_zero, tokenizer, test_prompts, device)
    else:
        print(f"Zero-error weights not found at {weights_path_zero}")

if __name__ == "__main__":
    main()
