import os
import sys
import torch
import torch.nn as nn
import time
import numpy as np

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Add the repo's library path to sys.path
sys.path.append(os.path.abspath("genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP

def gelu_exact(x):
    return 0.5 * x * (1.0 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))

class FusedGLUPolynomialCompiler:
    """
    Fused GLU Activation Compiler.
    Fuses both EML-KAN polynomial correction and GELU activation into a single unified polynomial:
    F(x) = GELU(x + P_eml(x)) \approx K_0 + K_1*x + K_2*x^2 + K_3*x^3
    Completely eliminates GELU math (tanh, cubics) from the runtime graph.
    """
    def __init__(self, model_layer, domain_bound=3.0):
        self.layer = model_layer
        self.domain_bound = domain_bound
        self.intermediate_size = model_layer.gate_proj.linear.out_features

    def fit_fused_activation(self, eml_poly_weights):
        print(f"Compiling and fusing GELU + EML into a single polynomial over range [-{self.domain_bound}, {self.domain_bound}]...")
        
        poly_p0 = eml_poly_weights["poly_p0"].numpy()
        poly_p1 = eml_poly_weights["poly_p1"].numpy()
        poly_p2 = eml_poly_weights["poly_p2"].numpy()
        poly_p3 = eml_poly_weights["poly_p3"].numpy()
        
        fused_k0 = np.zeros(self.intermediate_size, dtype=np.float32)
        fused_k1 = np.zeros(self.intermediate_size, dtype=np.float32)
        fused_k2 = np.zeros(self.intermediate_size, dtype=np.float32)
        fused_k3 = np.zeros(self.intermediate_size, dtype=np.float32)
        
        # Chebyshev nodes for minimax fitting
        cheb_nodes = np.cos((2 * np.arange(1, 15) - 1) * np.pi / 30) * self.domain_bound
        
        for i in range(self.intermediate_size):
            p0 = poly_p0[i]
            p1 = poly_p1[i]
            p2 = poly_p2[i]
            p3 = poly_p3[i]
            
            # Exact combined activation targets
            eml_corr = p0 + p1 * cheb_nodes + p2 * (cheb_nodes**2) + p3 * (cheb_nodes**3)
            gate_out = cheb_nodes + eml_corr
            f_targets = gelu_exact(gate_out)
            
            # Fit minimax 3rd-degree polynomial to the targets
            coeffs = np.polyfit(cheb_nodes, f_targets, 3)
            fused_k3[i], fused_k2[i], fused_k1[i], fused_k0[i] = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            
        print("GELU + EML fusion compiled successfully!")
        return {
            "k0": torch.tensor(fused_k0).float(),
            "k1": torch.tensor(fused_k1).float(),
            "k2": torch.tensor(fused_k2).float(),
            "k3": torch.tensor(fused_k3).float()
        }

class CompiledFusedGELUMLP(nn.Module):
    def __init__(self, base_mlp, fused_coeffs):
        super().__init__()
        self.gate_proj = base_mlp.gate_proj
        self.up_proj = base_mlp.up_proj
        self.down_proj = base_mlp.down_proj
        
        self.register_buffer("k0", fused_coeffs["k0"])
        self.register_buffer("k1", fused_coeffs["k1"])
        self.register_buffer("k2", fused_coeffs["k2"])
        self.register_buffer("k3", fused_coeffs["k3"])

    def forward(self, x):
        gate_linear = self.gate_proj(x)
        up_proj = self.up_proj(x)
        
        # Unified Single-Pass Fused Polynomial Activation (No GELU, no tanh!)
        x_squared = gate_linear * gate_linear
        x_cubed = x_squared * gate_linear
        
        activated = self.k0 + self.k1 * gate_linear + self.k2 * x_squared + self.k3 * x_cubed
        out = self.down_proj(activated * up_proj)
        return out
