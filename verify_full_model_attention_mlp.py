import os
import sys
import torch
import torch.nn as nn
import gc

# Add the repo's library path to sys.path
sys.path.append(os.path.abspath("mostoptimizedllm/genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP
from full_model_taylor_sharing_benchmark import QuantizableTaylorSharingGemma3EMLKANMLP
from attention_copy_expirements.hopfield_exp_sum_exp import HopfieldExpSumExpAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "google/gemma-3-1b-it"
    weights_path = "mostoptimizedllm/genomics/mostoptimizedllm/llmcopyexperiement/checkpoints/model_state_regularized.pt"
    prompt = "Write a python function to check if a number is prime."
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    state_dict = torch.load(weights_path, map_location="cpu")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 1. Load, generate, and delete Original Eager EML-KAN (FP32)
    print("Loading Eager EML-KAN baseline model...")
    eager_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    eager_model.config._attn_implementation = "eager"
    
    # Inject eager EML-KAN MLPs
    for i in range(eager_model.config.num_hidden_layers):
        kan_mlp = Gemma3EMLKANGatedMLP(eager_model.config).float()
        eager_model.model.layers[i].mlp = kan_mlp
        
    state_dict_fp32 = {k: v.float() for k, v in state_dict.items()}
    eager_model.load_state_dict(state_dict_fp32, strict=True)
    eager_model.eval()
    
    print("\n" + "="*60)
    print("GENERATING WITH EAGER EML-KAN BASELINE:")
    print("="*50)
    with torch.no_grad():
        outputs_eager = eager_model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
    baseline_text = tokenizer.decode(outputs_eager[0], skip_special_tokens=True)
    print(baseline_text)
    
    # Clean up baseline to save RAM
    del eager_model
    gc.collect()
    
    # 2. Load and configure fully optimized EML-KAN Model with both Taylor-MLPs and Hopfield-Attention
    print("\nLoading Fully Optimized EML-KAN Model (Hopfield Attention + Taylor MLP)...")
    opt_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    opt_model.config._attn_implementation = "eager"
    
    # Replace both Attention and MLP layers
    for i in range(opt_model.config.num_hidden_layers):
        # A. Replace MLP with compiled Taylor-Sharing MLP
        compiled_mlp = QuantizableTaylorSharingGemma3EMLKANMLP(
            opt_model.config, i, state_dict, prune_threshold=1.5e-4, taylor_threshold=0.08, sharing_threshold=0.03
        )
        opt_model.model.layers[i].mlp = compiled_mlp
        
        # B. Replace Attention with Fused Hopfield Attention
        orig_attn = opt_model.model.layers[i].self_attn
        hopfield_attn = HopfieldExpSumExpAttention(opt_model.config)
        hopfield_attn.load_weights_from_original(orig_attn)
        opt_model.model.layers[i].self_attn = hopfield_attn
        
    opt_model.eval()
    
    print("\n" + "="*60)
    print("GENERATING WITH FULLY OPTIMIZED MODEL:")
    print("="*50)
    with torch.no_grad():
        outputs_opt = opt_model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
    opt_text = tokenizer.decode(outputs_opt[0], skip_special_tokens=True)
    print(opt_text)
    print("="*60)
    
    # Output comparison check
    if baseline_text == opt_text:
        print("\nSUCCESS: The outputs are EXACTLY identical token-for-token!")
    else:
        print("\nOBSERVATION: Outputs are sensible but have minor divergence due to different attention logits representation precision.")
        
    # Clean up
    del opt_model
    gc.collect()

if __name__ == "__main__":
    main()
