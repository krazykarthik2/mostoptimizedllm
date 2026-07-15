import os
import torch
from model import Gemma3HopfieldKANAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "google/gemma-3-1b-it"
    device = "cuda:0"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    custom_prompt = (
        "Hello! I am John and I have 5 apples. If I give 2 apples to Mary and buy 3 "
        "more apples from the store, how many apples do I have now? Explain your reasoning step-by-step."
    )
    messages = [{"role": "user", "content": custom_prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    
    print("Swapping self_attn in every layer to Gemma3HopfieldKANAttention...")
    for i in range(model.config.num_hidden_layers):
        orig_attn = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = Gemma3HopfieldKANAttention(orig_attn).to(device)
        
    print("\nEvaluating the Modern Hopfield KAN Attention model on the reasoning prompt...")
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
    print("                      HOPFIELD KAN ATTENTION RESPONSE")
    print("="*80)
    print(response)
    print("="*80)

if __name__ == "__main__":
    main()
