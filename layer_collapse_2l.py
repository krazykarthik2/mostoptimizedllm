import os
import sys
import torch
import torch.nn as nn
import time
import numpy as np

# Adjust path to import model definitions
sys.path.append(os.path.abspath("genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP
from eml_hybrid_polynomial_compiler import EMLHybridPolynomialCompiler

class LayerCollapse2LCompiler:
    """
    2-Layer KAN Collapse Compiler.
    Fuses two consecutive KAN layers:
      y = f_2(f_1(x))
    where f_1 and f_2 are 3rd-degree polynomials.
    Since f_2(f_1(x)) results in a 9th-degree polynomial (causing parameter explosion),
    we dynamically approximate it back to a lower-degree minimax polynomial (e.g., degree 3 or 5)
    and prune terms with negligible coefficients to prevent parameter count explosion.
    """
    def __init__(self, layer1, layer2, domain_bound=3.0):
        self.layer1 = layer1
        self.layer2 = layer2
        self.domain_bound = domain_bound
        self.intermediate_size = layer1.gate_proj.linear.out_features

    def compile_collapsed_layers(self, prune_threshold=1.5e-4, taylor_threshold=0.08):
        print("Compiling consecutive layers for 2-layer KAN Collapse...")
        
        # 1. Compile both layers to polynomials
        comp1 = EMLHybridPolynomialCompiler(self.layer1, eps=1e-6)
        w_dict1 = comp1.fit_hybrid_polynomials(prune_threshold, taylor_threshold)
        
        comp2 = EMLHybridPolynomialCompiler(self.layer2, eps=1e-6)
        w_dict2 = comp2.fit_hybrid_polynomials(prune_threshold, taylor_threshold)
        
        # Load compiled polynomial coefficients
        p1_0 = w_dict1["poly_p0"].numpy()
        p1_1 = w_dict1["poly_p1"].numpy()
        p1_2 = w_dict1["poly_p2"].numpy()
        p1_3 = w_dict1["poly_p3"].numpy()
        
        p2_0 = w_dict2["poly_p0"].numpy()
        p2_1 = w_dict2["poly_p1"].numpy()
        p2_2 = w_dict2["poly_p2"].numpy()
        p2_3 = w_dict2["poly_p3"].numpy()
        
        # Chebyshev nodes for minimax evaluation
        cheb_nodes = np.cos((2 * np.arange(1, 15) - 1) * np.pi / 30) * self.domain_bound
        
        collapsed_p0 = np.zeros(self.intermediate_size, dtype=np.float32)
        collapsed_p1 = np.zeros(self.intermediate_size, dtype=np.float32)
        collapsed_p2 = np.zeros(self.intermediate_size, dtype=np.float32)
        collapsed_p3 = np.zeros(self.intermediate_size, dtype=np.float32)
        
        active_counts = 0
        
        for i in range(self.intermediate_size):
            # Layer 1 evaluation
            y1 = p1_0[i] + p1_1[i] * cheb_nodes + p1_2[i] * (cheb_nodes**2) + p1_3[i] * (cheb_nodes**3)
            # Layer 2 evaluation
            y2 = p2_0[i] + p2_1[i] * y1 + p2_2[i] * (y1**2) + p2_3[i] * (y1**3)
            
            # Fit combined target back to 3rd-degree minimax polynomial
            coeffs = np.polyfit(cheb_nodes, y2, 3)
            c3, c2, c1, c0 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            
            # Prune coefficients below threshold to avoid parameter explosion
            if abs(c3) < prune_threshold: c3 = 0.0
            if abs(c2) < prune_threshold: c2 = 0.0
            
            if c3 != 0.0 or c2 != 0.0:
                active_counts += 1
                
            collapsed_p0[i] = c0
            collapsed_p1[i] = c1
            collapsed_p2[i] = c2
            collapsed_p3[i] = c3
            
        print(f"2-Layer Collapse Complete: {active_counts}/{self.intermediate_size} active non-linear components retained.")
        
        return {
            "w_gate_linear": w_dict1["w_gate_linear"],
            "w_up": w_dict1["w_up"],
            "w_down": w_dict2["w_down"], # Out of layer 2
            "poly_p0": torch.tensor(collapsed_p0).float(),
            "poly_p1": torch.tensor(collapsed_p1).float(),
            "poly_p2": torch.tensor(collapsed_p2).float(),
            "poly_p3": torch.tensor(collapsed_p3).float(),
        }

def main():
    print("LayerCollapse2LCompiler module defined successfully!")

if __name__ == "__main__":
    main()
