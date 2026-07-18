import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add the repo's library path to sys.path using directory path traversal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mostoptimizedllm", "genomics", "mostoptimizedllm", "llmcopyexperiement")))
from model import Gemma3EMLKANAttention, EMLKANLinear
from transformers import AutoTokenizer, AutoModelForCausalLM

def algebraic_linear_to_kan_split(linear_layer, num_components=4, alpha=0.05):
    """
    Splits a standard linear layer projection W*x into a base linear path and an active
    EML KAN pathway algebraically, representing the target projection without training
    or zero-copying.
    
    y = W*x = W_base*x + EML(W_base*x)
    where W_base = (1 - alpha)*W, and EML path represents alpha*W*x.
    """
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    
    # Instantiate KAN Linear layer
    kan_linear = EMLKANLinear(in_features, out_features, num_components=num_components)
    
    # Extract original weight
    W = linear_layer.weight.data.clone()
    
    # 1. Base linear path weight
    W_base = (1.0 - alpha) * W
    kan_linear.linear.weight.data.copy_(W_base)
    
    # 2. EML path coefficients to represent gamma * z = (alpha / (1 - alpha)) * z
    gamma = alpha / (1.0 - alpha)
    
    # Parameters for strict zero-bias and exact derivative matching
    a_val = 1e-5
    c_val = 1e-5
    d_val = 2.0
    
    # Exact softplus and sigmoid at d = 2.0
    softplus_d = np.log(1.0 + np.exp(d_val))
    sigmoid_d = 1.0 / (1.0 + np.exp(-d_val))
    
    # Strict bias cancellation: exp(b) = log(softplus(d)) => b = log(log(softplus(d)))
    b_val = np.log(np.log(softplus_d))
    
    # Strict first derivative of exp(a*z + b) - log(softplus(c*z + d)) at z = 0 is:
    # L = a * exp(b) - c * (sigmoid(d) / softplus(d))
    coeff_factor = a_val * np.log(softplus_d) - c_val * (sigmoid_d / softplus_d)
    
    w_e_val = gamma / (coeff_factor * num_components)
    
    # Set parameters across all channels and components
    with torch.no_grad():
        kan_linear.eml.a.fill_(a_val)
        kan_linear.eml.b.fill_(b_val)
        kan_linear.eml.c.fill_(c_val)
        kan_linear.eml.d.fill_(d_val)
        kan_linear.eml.weight_eml.fill_(w_e_val)
        
    return kan_linear

def main():
    print("="*80)
    print("      ATTENTION ALGEBRAIC KAN SPLIT & MODERN HOPFIELD COMPILER")
    print("="*80)
    
    model_id = "google/gemma-3-1b-it"
    print("Loading original Gemma-3 model to extract attention weights...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    
    # Select Layer 0 attention
    orig_attn = model.model.layers[0].self_attn
    config = model.config
    
    print("\nCopying Attention projections using Algebraic Linear-to-KAN Split...")
    # Instantiate custom EML-KAN attention layer
    kan_attn = Gemma3EMLKANAttention(config, num_components=4)
    
    # Copy query, key, value, and output projections
    kan_attn.q_proj = algebraic_linear_to_kan_split(orig_attn.q_proj)
    kan_attn.k_proj = algebraic_linear_to_kan_split(orig_attn.k_proj)
    kan_attn.v_proj = algebraic_linear_to_kan_split(orig_attn.v_proj)
    kan_attn.o_proj = algebraic_linear_to_kan_split(orig_attn.o_proj)
    
    # Verify outputs of all projections directly
    print("\nVerifying output equivalence on a query input...")
    test_input = torch.randn(1, 10, config.hidden_size) # [batch, seq_len, hidden_size]
    
    # Q_dim for O proj input
    q_dim = config.num_attention_heads * config.head_dim
    test_input_o = torch.randn(1, 10, q_dim)
    
    with torch.no_grad():
        q_orig = orig_attn.q_proj(test_input)
        q_kan = kan_attn.q_proj(test_input)
        
        k_orig = orig_attn.k_proj(test_input)
        k_kan = kan_attn.k_proj(test_input)
        
        v_orig = orig_attn.v_proj(test_input)
        v_kan = kan_attn.v_proj(test_input)
        
        o_orig = orig_attn.o_proj(test_input_o)
        o_kan = kan_attn.o_proj(test_input_o)
        
    for name, orig, kan in [("Q Proj", q_orig, q_kan), 
                             ("K Proj", k_orig, k_kan), 
                             ("V Proj", v_orig, v_kan), 
                             ("O Proj", o_orig, o_kan)]:
        max_diff = torch.max(torch.abs(orig - kan)).item()
        mean_diff = torch.mean(torch.abs(orig - kan)).item()
        print(f"{name}: Max Diff = {max_diff:.2e}, Mean Diff = {mean_diff:.2e}")
        
    print("="*80)

if __name__ == "__main__":
    main()
