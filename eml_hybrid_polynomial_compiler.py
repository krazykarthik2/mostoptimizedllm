import os
import sys
import torch
import numpy as np
from scipy.optimize import curve_fit

# Adjust path to import model definitions
sys.path.append(os.path.abspath("genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP

def stable_softplus(x):
    return np.log(1.0 + np.exp(np.clip(x, -50.0, 20.0)))

def EML_exact(x, w_e, a, b, c, d, eps=1e-6):
    arg_x = np.clip(a * x + b, -10.0, 10.0)
    arg_y = c * x + d
    
    # Lossless asymptotic log-softplus
    log_softplus = np.where(
        arg_y > 20.0,
        np.log(arg_y),
        np.where(
            arg_y < -20.0,
            arg_y,
            np.log(stable_softplus(arg_y) + eps)
        )
    )
    return w_e * (np.exp(arg_x) - log_softplus)

class EMLHybridPolynomialCompiler:
    """
    Taylor-Polynomial Hybrid Compiler for EML KAN.
    Classifies each EML component into Taylor (linear), Asymptotic, or Chebyshev Minimax Polynomial
    regimes, then uses the distributive property to fold them into a single unified 3rd-degree polynomial.
    """
    def __init__(self, model_layer, eps=1e-6):
        self.layer = model_layer
        self.eps = eps
        self.hidden_size = model_layer.gate_proj.linear.in_features
        self.intermediate_size = model_layer.gate_proj.linear.out_features
        self.num_components = model_layer.gate_proj.eml.num_components

    def fit_hybrid_polynomials(self, prune_threshold=1.5e-4, taylor_threshold=0.08, domain_bound=10.0):
        print(f"Compiling layer with Taylor-Polynomial Hybrid Compiler (domain bounds: [-{domain_bound}, {domain_bound}])...")
        
        w_gate_linear = self.layer.gate_proj.linear.weight.detach().float().numpy()
        w_up = self.layer.up_proj.weight.detach().float().numpy()
        w_down = self.layer.down_proj.weight.detach().float().numpy()
        
        eml_a = self.layer.gate_proj.eml.a.detach().float().numpy()
        eml_b = self.layer.gate_proj.eml.b.detach().float().numpy()
        eml_c = self.layer.gate_proj.eml.c.detach().float().numpy()
        eml_d = self.layer.gate_proj.eml.d.detach().float().numpy()
        eml_w = self.layer.gate_proj.eml.weight_eml.detach().float().numpy()
        
        # Output coefficients per neuron
        poly_p0 = np.zeros(self.intermediate_size, dtype=np.float32)
        poly_p1 = np.zeros(self.intermediate_size, dtype=np.float32)
        poly_p2 = np.zeros(self.intermediate_size, dtype=np.float32)
        poly_p3 = np.zeros(self.intermediate_size, dtype=np.float32)
        
        # Grid for fitting Regime C components (Chebyshev nodes for minimax error minimization)
        cheb_nodes = np.cos((2 * np.arange(1, 11) - 1) * np.pi / 20) * domain_bound
        
        regime_counts = {"Taylor": 0, "Asymptotic": 0, "Chebyshev": 0}
        
        for i in range(self.intermediate_size):
            for k in range(self.num_components):
                w_e = eml_w[i, k]
                if abs(w_e) < prune_threshold:
                    continue
                
                a = eml_a[i, k]
                b = eml_b[i, k]
                c = eml_c[i, k]
                d = eml_d[i, k]
                
                # Check maximum bounds over domain
                max_arg_x = abs(a) * domain_bound + abs(b)
                max_arg_y = abs(c) * domain_bound + abs(d)
                
                # Regime A: Taylor Linearization (exact near zero)
                if max_arg_x < taylor_threshold and max_arg_y < taylor_threshold:
                    p0 = w_e * (1.3665 + b - 0.7213 * d)
                    p1 = w_e * (a - 0.7213 * c)
                    p2 = 0.0
                    p3 = 0.0
                    regime_counts["Taylor"] += 1
                    
                # Regime B: Asymptotic Constant / Linear
                elif (c * -domain_bound + d) < -20.0 and (c * domain_bound + d) < -20.0:
                    # softplus(v) \approx e^v, so log(softplus(v)) \approx v
                    # If exp part is also pruned or constant:
                    if (a * domain_bound + b) < -6.0:
                        p0 = -w_e * d
                        p1 = -w_e * c
                        p2 = 0.0
                        p3 = 0.0
                    else:
                        # Exp part remains, fit it with degree 3
                        ys = w_e * (np.exp(np.clip(a * cheb_nodes + b, -10.0, 10.0)) - (c * cheb_nodes + d))
                        coeffs = np.polyfit(cheb_nodes, ys, 3)
                        p3, p2, p1, p0 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                    regime_counts["Asymptotic"] += 1
                    
                # Regime C: Medium Non-linear Range (Chebyshev minimax fit)
                else:
                    ys = EML_exact(cheb_nodes, w_e, a, b, c, d, self.eps)
                    # Fit 3rd-degree polynomial to Chebyshev nodes
                    coeffs = np.polyfit(cheb_nodes, ys, 3)
                    p3, p2, p1, p0 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                    regime_counts["Chebyshev"] += 1
                
                # Accumulate pre-summed polynomial coefficients
                poly_p0[i] += p0
                poly_p1[i] += p1
                poly_p2[i] += p2
                poly_p3[i] += p3
                
        print(f"Hybrid compilation metrics:")
        print(f"  - Taylor (linearized): {regime_counts['Taylor']}")
        print(f"  - Asymptotic (linearized): {regime_counts['Asymptotic']}")
        print(f"  - Chebyshev (degree 3): {regime_counts['Chebyshev']}")
        
        weights_dict = {
            "w_gate_linear": torch.tensor(w_gate_linear).float(),
            "w_up": torch.tensor(w_up).float(),
            "w_down": torch.tensor(w_down).float(),
            "poly_p0": torch.tensor(poly_p0).float(),
            "poly_p1": torch.tensor(poly_p1).float(),
            "poly_p2": torch.tensor(poly_p2).float(),
            "poly_p3": torch.tensor(poly_p3).float(),
        }
        return weights_dict
