import os
import torch
from model import Gemma3HopfieldKANAttention, Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_zero_error_eval():
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
    
    print("Swapping self_attn and mlp layers to Hopfield-KAN and Gated-EML-KAN...")
    for i in range(model.config.num_hidden_layers):
        # 1. Swap Attention
        orig_attn = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = Gemma3HopfieldKANAttention(orig_attn).to(device)
        
        # 2. Swap MLP and copy weights 1:1
        orig_mlp = model.model.layers[i].mlp
        kan_mlp = Gemma3EMLKANGatedMLP(model.config).to(torch.bfloat16).to(device)
        
        with torch.no_grad():
            # Copy linear projection weights
            kan_mlp.gate_proj.linear.weight.copy_(orig_mlp.gate_proj.weight)
            kan_mlp.up_proj.weight.copy_(orig_mlp.up_proj.weight)
            kan_mlp.down_proj.weight.copy_(orig_mlp.down_proj.weight)
            
        model.model.layers[i].mlp = kan_mlp
        
    print("\nEvaluating the Zero-Error EML-KAN model on the OOD reasoning prompt...")
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
    print("                      ZERO-ERROR EML-KAN HYBRID RESPONSE")
    print("="*80)
    print(response)
    print("="*80)

if __name__ == "__main__":
    run_zero_error_eval()
