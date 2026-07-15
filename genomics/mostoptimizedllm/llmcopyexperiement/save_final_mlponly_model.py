import os
import torch
from model import Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "google/gemma-3-1b-it"
    save_path = "gemma3_eml_kan/model_state_final_mlponly.pt"
    device = "cuda:0"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    test_prompts = [
        "If a train travels 60 miles per hour, how far will it travel in 2.5 hours? Explain your reasoning step-by-step.",
        "A father has 4 daughters. Each daughter has a brother. How many children does the father have in total? Explain your reasoning.",
        "Write a python function to find the largest element in a list of numbers.",
        "What is the largest ocean on Earth?"
    ]
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    
    print("Swapping ONLY MLP layers to Gemma3EMLKANGatedMLP (Attention kept native)...")
    for i in range(model.config.num_hidden_layers):
        orig_mlp = model.model.layers[i].mlp
        kan_mlp = Gemma3EMLKANGatedMLP(model.config).to(torch.bfloat16).to(device)
        with torch.no_grad():
            kan_mlp.gate_proj.linear.weight.copy_(orig_mlp.gate_proj.weight)
            kan_mlp.up_proj.weight.copy_(orig_mlp.up_proj.weight)
            kan_mlp.down_proj.weight.copy_(orig_mlp.down_proj.weight)
        model.model.layers[i].mlp = kan_mlp
        
    print(f"Saving optimal MLP-only KAN hybrid state dict to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    print("\nRunning validation checks on the saved configuration...")
    for p in test_prompts:
        messages = [{"role": "user", "content": p}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print("\n" + "="*80)
        print(f"Prompt: {p}")
        print("="*80)
        print(response)
        print("="*80)

if __name__ == "__main__":
    main()
