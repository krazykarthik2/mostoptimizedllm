import os
import json
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
from model import Gemma3EMLKANGatedMLP
from transformers import AutoConfig

# ==============================================================================
# 1. EML Grammar Symbolic Functions
# ==============================================================================

# Simple Symbolic grammar candidates to replace exp/log
def grammar_identity(x):
    return x

def grammar_shift(x, c):
    return x + c

def grammar_scale(x, w):
    return w * x

def grammar_poly2(x, w, c):
    return w * (x ** 2) + c

GRAMMAR_FUNCS = {
    "identity": (grammar_identity, 0, 1.0),      # (func, num_params, speed_multiplier)
    "shift": (grammar_shift, 1, 1.2),
    "scale": (grammar_scale, 1, 1.2),
    "poly2": (grammar_poly2, 2, 2.0)
}

# ==============================================================================
# 2. Symbolic DAG Optimizer
# ==============================================================================

class EMLSymbolicDAGOptimizer:
    def __init__(self, model_config, weights_path, mse_threshold=1e-3):
        self.config = model_config
        self.weights_path = weights_path
        self.mse_threshold = mse_threshold
        
        # Instantiate model structure
        self.device = "cpu"
        self.layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        
        print("Loading trained KAN weights from disk...")
        self.state_dict = torch.load(weights_path, map_location=self.device)
        
    def fit_symbolic_grammar(self, x_grid, y_curve):
        """
        Attempts to fit simpler EML grammar functions to the learned curve.
        Returns the best function name, fitted parameters, and MSE.
        """
        best_func = "eml_original"  # Fallback (keeps original exp/log)
        best_params = []
        best_mse = np.mean(y_curve ** 2)  # Base MSE against zero
        
        # Test Identity fit (no params)
        y_fit = grammar_identity(x_grid)
        mse = np.mean((y_curve - y_fit) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_func = "identity"
            best_params = []
            
        # Test Shift fit (1 param)
        try:
            popt, _ = curve_fit(grammar_shift, x_grid, y_curve, p0=[0.0], maxfev=1000)
            y_fit = grammar_shift(x_grid, *popt)
            mse = np.mean((y_curve - y_fit) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_func = "shift"
                best_params = popt.tolist()
        except:
            pass
            
        # Test Scale fit (1 param)
        try:
            popt, _ = curve_fit(grammar_scale, x_grid, y_curve, p0=[1.0], maxfev=1000)
            y_fit = grammar_scale(x_grid, *popt)
            mse = np.mean((y_curve - y_fit) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_func = "scale"
                best_params = popt.tolist()
        except:
            pass
            
        # Test Poly2 fit (2 params)
        try:
            popt, _ = curve_fit(grammar_poly2, x_grid, y_curve, p0=[1.0, 0.0], maxfev=1000)
            y_fit = grammar_poly2(x_grid, *popt)
            mse = np.mean((y_curve - y_fit) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_func = "poly2"
                best_params = popt.tolist()
        except:
            pass
            
        return best_func, best_params, best_mse

    def optimize_network_to_dag(self):
        """
        Decomposes active EML curves into a symbolic DAG representation.
        Applies Speed vs Size Pareto selection.
        """
        dag_structure = {
            "model_metadata": {
                "layers": self.layers,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size
            },
            "dag_layers": {}
        }
        
        # Grid range to evaluate functions over
        x_grid = np.linspace(-2.0, 2.0, 200)
        
        total_edges = 0
        pruned_edges = 0
        simplified_edges = 0
        
        print("\nStarting EML Grammar Decomposition & Pareto Selection...")
        
        for layer_idx in range(self.layers):
            layer_key_prefix = f"model.layers.{layer_idx}.mlp.gate_proj.eml."
            
            # Extract EML parameters for this layer
            a = self.state_dict[layer_key_prefix + "a"].float().numpy() # [intermediate_size, num_components]
            b = self.state_dict[layer_key_prefix + "b"].float().numpy()
            c = self.state_dict[layer_key_prefix + "c"].float().numpy()
            d = self.state_dict[layer_key_prefix + "d"].float().numpy()
            weight_eml = self.state_dict[layer_key_prefix + "weight_eml"].float().numpy()
            
            num_channels, num_components = weight_eml.shape
            layer_dag = []
            
            for channel in range(num_channels):
                channel_dag_nodes = []
                
                for comp in range(num_components):
                    total_edges += 1
                    w_eml = weight_eml[channel, comp]
                    
                    # 1. Sparsity pruning: if contribution is near zero, prune the edge completely!
                    if abs(w_eml) < 1e-4:
                        pruned_edges += 1
                        continue
                        
                    # Compute continuous learned curve
                    comp_a = a[channel, comp]
                    comp_b = b[channel, comp]
                    comp_c = c[channel, comp]
                    comp_d = d[channel, comp]
                    
                    # arg_x and arg_y calculations
                    arg_x = np.clip(comp_a * x_grid + comp_b, -10.0, 10.0)
                    arg_y = np.log(1.0 + np.exp(comp_c * x_grid + comp_d)) + 1e-6 # Softplus
                    y_curve = w_eml * (np.exp(arg_x) - np.log(arg_y))
                    
                    # 2. Fit grammar functions
                    best_func, best_params, best_mse = self.fit_symbolic_grammar(x_grid, y_curve)
                    
                    # 3. Pareto selection: Accept simplified function if MSE is below threshold
                    if best_mse <= self.mse_threshold and best_func != "eml_original":
                        simplified_edges += 1
                        node_op = best_func
                        node_params = best_params
                    else:
                        # Fallback to original continuous EML (retains full accuracy)
                        node_op = "eml_original"
                        node_params = [float(comp_a), float(comp_b), float(comp_c), float(comp_d), float(w_eml)]
                        
                    channel_dag_nodes.append({
                        "component_idx": comp,
                        "op": node_op,
                        "params": node_params,
                        "mse": float(best_mse)
                    })
                    
                if len(channel_dag_nodes) > 0:
                    layer_dag.append({
                        "channel_idx": channel,
                        "operations": channel_dag_nodes
                    })
                    
            dag_structure["dag_layers"][f"layer_{layer_idx}"] = layer_dag
            print(f"  Layer {layer_idx:2d} Optimized: {len(layer_dag)} active channels.")
            
        print("\n" + "="*80)
        print("                      DAG OPTIMIZER SUMMARY")
        print("="*80)
        print(f"Total evaluated KAN edges: {total_edges}")
        print(f"Pruned inactive edges (Sparsity): {pruned_edges} ({pruned_edges/total_edges*100:.2f}%)")
        print(f"Decomposed to simpler grammar functions (+, -, scale): {simplified_edges} ({simplified_edges/total_edges*100:.2f}%)")
        print(f"Retained as complex EML curves (Accuracy fallback): {total_edges - pruned_edges - simplified_edges}")
        print("="*80)
        
        return dag_structure

def main():
    config_path = "gemma3_eml_kan/config.json"
    weights_path = "gemma3_eml_kan/model_state_regularized.pt"
    output_dag_path = "gemma3_eml_kan/symbolic_dag_model.json"
    
    model_id = "google/gemma-3-1b-it"
    # Load configuration
    config = AutoConfig.from_pretrained(model_id)
        
    optimizer = EMLSymbolicDAGOptimizer(config, weights_path, mse_threshold=1e-3)
    dag_model = optimizer.optimize_network_to_dag()
    
    print(f"Saving compiled Symbolic DAG representation to {output_dag_path}...")
    with open(output_dag_path, "w") as f:
        json.dump(dag_model, f, indent=2)
        
    print("Decomposition completed successfully!")

if __name__ == "__main__":
    main()
