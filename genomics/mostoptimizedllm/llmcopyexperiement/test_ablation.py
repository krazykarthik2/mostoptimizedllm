import os
import torch
from model import Gemma3HopfieldKANAttention, Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_evaluation_ablation(model, tokenizer, test_prompts, device):
    results = []
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
        results.append((p, response))
    return results

def main():
    model_id = "google/gemma-3-1b-it"
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    test_prompts = [
        "If a train travels 60 miles per hour, how far will it travel in 2.5 hours? Explain your reasoning step-by-step.",
        "A father has 4 daughters. Each daughter has a brother. How many children does the father have in total? Explain your reasoning."
    ]
    
    # --------------------------------------------------------------------------
    # Test Ablation A: MLP Swap Only (Original Attention kept)
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print(" ABLATION A: MLP Swap Only (Original Attention kept)")
    print("="*80)
    model_a = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model_a.eval()
    
    for i in range(model_a.config.num_hidden_layers):
        orig_mlp = model_a.model.layers[i].mlp
        kan_mlp = Gemma3EMLKANGatedMLP(model_a.config).to(torch.bfloat16).to(device)
        with torch.no_grad():
            kan_mlp.gate_proj.linear.weight.copy_(orig_mlp.gate_proj.weight)
            kan_mlp.up_proj.weight.copy_(orig_mlp.up_proj.weight)
            kan_mlp.down_proj.weight.copy_(orig_mlp.down_proj.weight)
        model_a.model.layers[i].mlp = kan_mlp
        
    results_a = run_evaluation_ablation(model_a, tokenizer, test_prompts, device)
    for p, r in results_a:
        print(f"\nPrompt: {p}\nResponse:\n{r}")
        print("-"*80)
        
    del model_a
    torch.cuda.empty_cache()
    
    # --------------------------------------------------------------------------
    # Test Ablation B: Attention Swap Only (Original MLP kept)
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print(" ABLATION B: Attention Swap Only (Original MLP kept)")
    print("="*80)
    model_b = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model_b.eval()
    
    for i in range(model_b.config.num_hidden_layers):
        orig_attn = model_b.model.layers[i].self_attn
        model_b.model.layers[i].self_attn = Gemma3HopfieldKANAttention(orig_attn).to(device)
        
    results_b = run_evaluation_ablation(model_b, tokenizer, test_prompts, device)
    for p, r in results_b:
        print(f"\nPrompt: {p}\nResponse:\n{r}")
        print("-"*80)

if __name__ == "__main__":
    main()
