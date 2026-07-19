import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Adjust path to import model definitions
sys.path.append(os.path.abspath("mostoptimizedllm/genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP
from eml_hybrid_polynomial_compiler import EMLHybridPolynomialCompiler

def fit_composite_polynomial(poly_list, domain_bound=3.0, prune_threshold=1.5e-4):
    """
    Fits a sequence of polynomials (representing consecutive KAN layers) into a single 3rd-degree polynomial.
    """
    cheb_nodes = np.cos((2 * np.arange(1, 15) - 1) * np.pi / 30) * domain_bound
    intermediate_size = len(poly_list[0]["poly_p0"])
    
    collapsed_p0 = np.zeros(intermediate_size, dtype=np.float32)
    collapsed_p1 = np.zeros(intermediate_size, dtype=np.float32)
    collapsed_p2 = np.zeros(intermediate_size, dtype=np.float32)
    collapsed_p3 = np.zeros(intermediate_size, dtype=np.float32)
    
    for i in range(intermediate_size):
        y = cheb_nodes
        for poly in poly_list:
            p0 = poly["poly_p0"][i]
            p1 = poly["poly_p1"][i]
            p2 = poly["poly_p2"][i]
            p3 = poly["poly_p3"][i]
            y = p0 + p1 * y + p2 * (y**2) + p3 * (y**3)
            
        coeffs = np.polyfit(cheb_nodes, y, 3)
        c3, c2, c1, c0 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        
        if abs(c3) < prune_threshold: c3 = 0.0
        if abs(c2) < prune_threshold: c2 = 0.0
        
        collapsed_p0[i] = c0
        collapsed_p1[i] = c1
        collapsed_p2[i] = c2
        collapsed_p3[i] = c3
        
    return {
        "poly_p0": collapsed_p0,
        "poly_p1": collapsed_p1,
        "poly_p2": collapsed_p2,
        "poly_p3": collapsed_p3,
    }

class EMLDPCollapseCompiler:
    """
    Dynamic Programming (DP) KAN Layer Collapse Compiler.
    Finds the optimal partitioning of consecutive KAN layers to maximize speed (minimizing layers)
    while keeping the accumulated approximation drift below a strict threshold.
    Supports 1-layer, 2-layer, and 3-layer merges.
    """
    def __init__(self, config, state_dict, max_layers=26, error_threshold=1.5e-2, lambd=1e3):
        self.config = config
        self.state_dict = state_dict
        self.max_layers = max_layers
        self.error_threshold = error_threshold
        self.lambd = lambd # Trade-off weight between speed and accuracy
        
    def evaluate_merge_error(self, start_idx, end_idx):
        """
        Evaluates the approximation error of collapsing layers from start_idx to end_idx (inclusive).
        """
        num_layers_to_merge = end_idx - start_idx + 1
        if num_layers_to_merge == 1:
            return 0.0, None # No merge, zero error
            
        # Compile each layer to polynomials first
        poly_list = []
        for idx in range(start_idx, end_idx + 1):
            dummy = Gemma3EMLKANGatedMLP(self.config, num_components=4)
            d_state = {}
            for k, v in self.state_dict.items():
                if f"model.layers.{idx}.mlp." in k:
                    d_state[k.replace(f"model.layers.{idx}.mlp.", "")] = v
            dummy.load_state_dict(d_state)
            dummy.eval()
            
            compiler = EMLHybridPolynomialCompiler(dummy, eps=1e-6)
            w_dict = compiler.fit_hybrid_polynomials(prune_threshold=1.5e-4, taylor_threshold=0.08)
            poly_list.append({
                "poly_p0": w_dict["poly_p0"].numpy(),
                "poly_p1": w_dict["poly_p1"].numpy(),
                "poly_p2": w_dict["poly_p2"].numpy(),
                "poly_p3": w_dict["poly_p3"].numpy(),
            })
            
        # Fit collapsed composite polynomial
        collapsed = fit_composite_polynomial(poly_list, domain_bound=3.0, prune_threshold=1.5e-4)
        
        # Calculate approximation drift on Chebyshev nodes
        cheb_nodes = np.cos((2 * np.arange(1, 15) - 1) * np.pi / 30) * 3.0
        intermediate_size = len(poly_list[0]["poly_p0"])
        
        errors = []
        for i in range(intermediate_size):
            # Target exact sequential evaluation
            y_target = cheb_nodes
            for poly in poly_list:
                y_target = poly["poly_p0"][i] + poly["poly_p1"][i] * y_target + poly["poly_p2"][i] * (y_target**2) + poly["poly_p3"][i] * (y_target**3)
                
            # Collapsed evaluation
            y_collapsed = collapsed["poly_p0"][i] + collapsed["poly_p1"][i] * cheb_nodes + collapsed["poly_p2"][i] * (cheb_nodes**2) + collapsed["poly_p3"][i] * (cheb_nodes**3)
            
            mae = np.mean(np.abs(y_target - y_collapsed))
            errors.append(mae)
            
        mean_error = float(np.mean(errors))
        return mean_error, collapsed

    def search_optimal_collapses(self):
        L = self.max_layers
        print(f"\nRunning DP Layer Collapse search over {L} layers...")
        
        # DP state: dp[i] = (min_cost, path_backpointer, accumulated_error)
        dp = [(999.0, -1, 0.0)] * (L + 1)
        dp[0] = (0.0, -1, 0.0) # Base case
        
        # Keep track of computed collapses to avoid redundant work
        merge_results = {}
        
        for i in range(1, L + 1):
            for step in [1, 2, 3]:
                k = i - step
                if k >= 0:
                    prev_cost, _, prev_err = dp[k]
                    if prev_cost != 999.0:
                        # Evaluate merge error
                        err, collapsed_w = self.evaluate_merge_error(k, i - 1)
                        total_err = prev_err + err
                        
                        # Check if error bounds are satisfied
                        if err <= self.error_threshold:
                            # Unified cost function: speedup (1 / step) + trade-off * error
                            cost = prev_cost + (1.0 / step) + (self.lambd * err)
                            if cost < dp[i][0]:
                                dp[i] = (cost, k, total_err)
                                if collapsed_w is not None:
                                    merge_results[(k, i - 1)] = collapsed_w
                                    
        # Backtrack optimal partition path
        path = []
        curr = L
        while curr > 0:
            prev = dp[curr][1]
            path.append((prev, curr - 1))
            curr = prev
        path.reverse()
        
        print("\nOptimal DP Layer Partitioning Decided:")
        for start, end in path:
            size = end - start + 1
            print(f"  - Layers {start} to {end}: Block size {size} (Error: {dp[end+1][2] - dp[start][2]:.4f})")
            
        return path, merge_results
