import os
import torch
from model import Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state_regularized.pt"
    device = "cuda:0"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    test_prompts = [
        "If a rectangle has a length of 12 cm and a width of 5 cm, what is its perimeter? Explain step-by-step.",
        "If all dogs are animals, and some animals are furry, can we conclude that all dogs are furry? Explain your reasoning.",
        "Write a python function to check if a given integer is a prime number.",
        "What is the height of Mount Everest?"
    ]
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    
    print("Swapping MLP layers to Gemma3EMLKANGatedMLP...")
    for i in range(model.config.num_hidden_layers):
        kan_mlp = Gemma3EMLKANGatedMLP(model.config).to(torch.bfloat16).to(device)
        model.model.layers[i].mlp = kan_mlp
        
    print(f"Loading regularized weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    print("\nEvaluating the regularized model on new questions...")
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
