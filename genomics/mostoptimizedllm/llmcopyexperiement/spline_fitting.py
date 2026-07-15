import os
import gc
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from model import Gemma3EMLKANMLP, EMLKANLinear
from transformers import AutoConfig, AutoModelForCausalLM

def compute_cosine_similarity(a, b):
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    return F.cosine_similarity(a_flat, b_flat, dim=0).item()

def fit_layer(layer_idx, x_path, y_path, config, original_mlp=None, device="cuda"):
    print(f"\n--- Fitting Layer {layer_idx} ---")
    
    # Load farmed activation tensors
    x = torch.load(x_path).to(device).to(torch.float32)
    y = torch.load(y_path).to(device).to(torch.float32)
    
    # Initialize the custom EML-KAN MLP block
    kan_mlp = Gemma3EMLKANMLP(config).to(device).to(torch.float32)
    
    # 1. Initialize linear weights using original MLP's projection weights if available
    if original_mlp is not None:
        # original_mlp has gate_proj, up_proj, down_proj
        # In GeGLU: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        # We can initialize KAN linear weight 1 with up_proj weight, and weight 2 with down_proj weight
        with torch.no_grad():
            kan_mlp.ffn1.linear.weight.copy_(original_mlp.up_proj.weight.data)
            kan_mlp.ffn2.linear.weight.copy_(original_mlp.down_proj.weight.data)
            print("Initialized linear weights from original MLP projections.")
            
    # 2. Dynamic Boundary Adaptation
    # Compute inputs to the activation layers to find coordinate min/max ranges
    with torch.no_grad():
        z1 = kan_mlp.ffn1.linear(x) # [N, intermediate_size]
        
    # Scale and bias EML-KAN activations dynamically for ffn1
    # Z1 boundaries
    z1_min, _ = torch.min(z1, dim=0)
    z1_max, _ = torch.max(z1, dim=0)
    
    # Set KAN activation parameters based on range
    with torch.no_grad():
        # Map input z1 to safe region [-2, 2] for the exponential pathway
        eps = 1e-5
        range_z1 = z1_max - z1_min + eps
        
        # self.a: [channels, num_components], self.b: [channels, num_components]
        for k in range(kan_mlp.ffn1.act.num_components):
            kan_mlp.ffn1.act.a[:, k] = 4.0 / range_z1
            kan_mlp.ffn1.act.b[:, k] = -2.0 * (z1_max + z1_min) / range_z1
            
            # Map input z1 to positive region [1.1, 11.0] for the logarithm pathway
            kan_mlp.ffn1.act.c[:, k] = 9.9 / range_z1
            kan_mlp.ffn1.act.d[:, k] = 1.1 - kan_mlp.ffn1.act.c[:, k] * z1_min
            
        # Initialize base residual weight to 1.0, and EML weight to a small value
        kan_mlp.ffn1.act.weight_base.fill_(1.0)
        kan_mlp.ffn1.act.weight_eml.fill_(0.01)

    # Compute intermediate features H
    with torch.no_grad():
        h = kan_mlp.ffn1(x)
        z2 = kan_mlp.ffn2.linear(h)
        
    # Scale and bias EML-KAN activations dynamically for ffn2
    z2_min, _ = torch.min(z2, dim=0)
    z2_max, _ = torch.max(z2, dim=0)
    
    with torch.no_grad():
        range_z2 = z2_max - z2_min + eps
        for k in range(kan_mlp.ffn2.act.num_components):
            kan_mlp.ffn2.act.a[:, k] = 4.0 / range_z2
            kan_mlp.ffn2.act.b[:, k] = -2.0 * (z2_max + z2_min) / range_z2
            
            kan_mlp.ffn2.act.c[:, k] = 9.9 / range_z2
            kan_mlp.ffn2.act.d[:, k] = 1.1 - kan_mlp.ffn2.act.c[:, k] * z2_min
            
        kan_mlp.ffn2.act.weight_base.fill_(1.0)
        kan_mlp.ffn2.act.weight_eml.fill_(0.01)

    # Calculate initial MSE and Cosine Similarity
    with torch.no_grad():
        pred_init = kan_mlp(x)
        loss_init = F.mse_loss(pred_init, y).item()
        cos_init = compute_cosine_similarity(pred_init, y)
        print(f"Initial L2 Loss (MSE): {loss_init:.6f} | Cosine Similarity: {cos_init:.4f}")

    # 3. L-BFGS Parameter Optimization
    # Fit the KAN module to map x -> y
    # Sample a representative subset to make L-BFGS fast and memory efficient
    fit_samples = min(4096, x.shape[0])
    indices = torch.randperm(x.shape[0])[:fit_samples]
    x_batch = x[indices]
    y_batch = y[indices]

    # Optimize KAN weights
    optimizer = torch.optim.LBFGS(
        kan_mlp.parameters(), 
        lr=0.1, 
        max_iter=80, 
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        preds = kan_mlp(x_batch)
        loss = F.mse_loss(preds, y_batch)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Calculate final MSE and Cosine Similarity
    with torch.no_grad():
        pred_final = kan_mlp(x)
        loss_final = F.mse_loss(pred_final, y).item()
        cos_final = compute_cosine_similarity(pred_final, y)
        print(f"Final L2 Loss (MSE): {loss_final:.6f} | Cosine Similarity: {cos_final:.4f}")
        
    return kan_mlp.cpu()

def fit_and_hot_swap_model(model_id, farmed_dir, save_path, device="cuda"):
    print("Loading base model configuration and weights...")
    config = AutoConfig.from_pretrained(model_id)
    # Load model in CPU/meta/GPU depending on constraints; here we load it on GPU to extract MLP weights
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float16
    ).to(device)
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    
    for i in range(num_layers):
        x_path = os.path.join(farmed_dir, f"x_layer_{i}.pt")
        y_path = os.path.join(farmed_dir, f"y_layer_{i}.pt")
        
        # Get the original MLP block
        original_mlp = model.model.layers[i].mlp
        
        # Fit the EML-KAN module
        fitted_kan = fit_layer(i, x_path, y_path, config, original_mlp, device=device)
        
        # Perform structural hot-swap: excise original MLP, insert KAN MLP
        # Cast to model's default dtype (float16/bfloat16) before swapping
        fitted_kan = fitted_kan.to(model.dtype)
        model.model.layers[i].mlp = fitted_kan
        
        # Clear CUDA memory and garbage collect
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll layers hot-swapped successfully with fitted EML-KAN blocks.")
    
    # Save the modified model
    print(f"Saving new EML-KAN model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model_state.pt"))
    # Also save config so it can be loaded
    config.save_pretrained(save_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B: Spline Fitting & Structural Hot-Swapping")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="Original Gemma model ID")
    parser.add_argument("--farmed_dir", type=str, default="farmed_activations", help="Directory containing farmed activations")
    parser.add_argument("--save_path", type=str, default="gemma3_eml_kan", help="Path to save the new model")
    args = parser.parse_args()
    
    fit_and_hot_swap_model(args.model_id, args.farmed_dir, args.save_path)
