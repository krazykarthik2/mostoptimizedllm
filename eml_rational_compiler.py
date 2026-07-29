import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath("genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP

def stable_softplus(x):
    return np.log(1.0 + np.exp(np.clip(x, -50.0, 20.0)))

def EML_A(x, w_e, a, b):
    arg_x = 3.0 * np.tanh((a * x + b) / 3.0)
    return w_e * np.exp(arg_x)

def EML_B(x, w_e, c, d, eps=1e-6):
    arg_y = c * x + d
    log_softplus = np.where(
        arg_y > 20.0,
        np.log(arg_y),
        np.where(
            arg_y < -20.0,
            arg_y,
            np.log(stable_softplus(arg_y) + eps)
        )
    )
    return -w_e * log_softplus

def EML_exact(x, w_e, a, b, c, d, eps=1e-6):
    return EML_A(x, w_e, a, b) + EML_B(x, w_e, c, d, eps)

def rational_pade_1_1(x, p0, p1, q1):
    return (p0 + p1 * x) / (1.0 + np.abs(q1 * x))

class EMLRationalCompiler:
    """
    Padé Rational [1/1] Compiler for EML KAN.
    Approximates the EML correction term using a highly stable rational function:
    f(x) \approx (p0 + p1 * x) / (1 + |q1 * x|)
    """
    def __init__(self, model_layer, eps=1e-6):
        self.layer = model_layer
        self.eps = eps
        self.hidden_size = model_layer.gate_proj.linear.in_features
        self.intermediate_size = model_layer.gate_proj.linear.out_features
        self.num_components = model_layer.gate_proj.eml.num_components

    def fit_rational_approximations(self, prune_threshold=1.5e-4, domain_bound=3.0):
        print(f"Compiling layer with Padé [1/1] Rational Compiler (domain bounds: [-{domain_bound}, {domain_bound}])...")
        
        eml_a = self.layer.gate_proj.eml.a.detach().float().numpy()
        eml_b = self.layer.gate_proj.eml.b.detach().float().numpy()
        eml_c = self.layer.gate_proj.eml.c.detach().float().numpy()
        eml_d = self.layer.gate_proj.eml.d.detach().float().numpy()
        eml_w = self.layer.gate_proj.eml.weight_eml.detach().float().numpy()
        
        poly_p0 = np.zeros(self.intermediate_size, dtype=np.float32)
        poly_p1 = np.zeros(self.intermediate_size, dtype=np.float32)
        poly_q1 = np.zeros(self.intermediate_size, dtype=np.float32)
        
        cheb_nodes = np.cos((2 * np.arange(1, 11) - 1) * np.pi / 20) * domain_bound
        
        for i in range(self.intermediate_size):
            ys_total = np.zeros_like(cheb_nodes)
            for k in range(self.num_components):
                w_e = eml_w[i, k]
                if abs(w_e) < prune_threshold:
                    continue
                ys_total += EML_exact(cheb_nodes, w_e, eml_a[i, k], eml_b[i, k], eml_c[i, k], eml_d[i, k])
            
            try:
                popt, _ = curve_fit(rational_pade_1_1, cheb_nodes, ys_total, p0=[0.0, 0.0, 0.0], maxfev=1000)
                poly_p0[i], poly_p1[i], poly_q1[i] = popt[0], popt[1], popt[2]
            except Exception:
                # Fallback to linear Taylor
                poly_p0[i] = ys_total[5]
                poly_p1[i] = (ys_total[-1] - ys_total[0]) / (2 * domain_bound)
                poly_q1[i] = 0.0
                
        return {
            "poly_p0": torch.tensor(poly_p0),
            "poly_p1": torch.tensor(poly_p1),
            "poly_q1": torch.tensor(poly_q1),
        }
