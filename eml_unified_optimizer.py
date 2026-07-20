import os
import sys
import time
import torch
import sympy as sp
import numpy as np

# Adjust path to import model definitions from the repository
sys.path.append(os.path.abspath("genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP

class EMLKANUnifiedCompiler:
    """
    Optimal Compiler for EML KAN DAGs.
    Converts a Gemma-3 EML-KAN MLP layer into a highly optimized, fully vectorized
    NumPy-executable DAG. Incorporates constant precomputation and folding.
    """
    def __init__(self, model_layer, eps=1e-6):
        self.layer = model_layer
        self.eps = eps
        
        self.hidden_size = model_layer.gate_proj.linear.in_features
        self.intermediate_size = model_layer.gate_proj.linear.out_features
        self.num_components = model_layer.gate_proj.eml.num_components

    def compile_layer(self, prune_threshold=1.5e-4, constant_threshold=1e-3):
        """
        Compiles the Gemma3EMLKANGatedMLP layer to a highly optimized vectorized NumPy function
        with threshold-based KAN component pruning and constant precomputation/folding.
        """
        print(f"Extracting weights and building Sparse Vectorized DAG (threshold: {prune_threshold})...")
        print(f"Constant precomputation active (threshold: {constant_threshold})...")
        
        # Get weight matrices as numpy arrays
        w_gate_linear = self.layer.gate_proj.linear.weight.detach().float().numpy()  # [intermediate_size, hidden_size]
        w_up = self.layer.up_proj.weight.detach().float().numpy()                    # [intermediate_size, hidden_size]
        w_down = self.layer.down_proj.weight.detach().float().numpy()                # [hidden_size, intermediate_size]
        
        # EML params
        eml_a = self.layer.gate_proj.eml.a.detach().float().numpy()                  # [intermediate_size, num_components]
        eml_b = self.layer.gate_proj.eml.b.detach().float().numpy()                  # [intermediate_size, num_components]
        eml_c = self.layer.gate_proj.eml.c.detach().float().numpy()                  # [intermediate_size, num_components]
        eml_d = self.layer.gate_proj.eml.d.detach().float().numpy()                  # [intermediate_size, num_components]
        eml_w = self.layer.gate_proj.eml.weight_eml.detach().float().numpy()         # [intermediate_size, num_components]
        
        # 1. Precompute constants where parameters are close to zero
        # If abs(a) < constant_threshold, exp(a*x + b) is constant and equals exp(b)
        # If abs(c) < constant_threshold, log(softplus(c*x + d) + eps) is constant and equals log(softplus(d) + eps)
        
        # Precomputed constant arrays
        const_exp_vals = np.zeros_like(eml_a)
        const_log_vals = np.zeros_like(eml_c)
        const_full_eml = np.zeros(self.intermediate_size, dtype=np.float32)
        
        # Mask of where we precompute
        mask_a_const = np.abs(eml_a) < constant_threshold
        mask_c_const = np.abs(eml_c) < constant_threshold
        
        # Softplus function helper for constant calculation
        def softplus(x):
            return np.log(1.0 + np.exp(np.clip(x, -50.0, 20.0)))
            
        # Calculate precomputed values
        const_exp_vals[mask_a_const] = np.exp(eml_b[mask_a_const])
        const_log_vals[mask_c_const] = np.log(softplus(eml_d[mask_c_const]) + self.eps)
        
        # If BOTH are constant, we can fold the entire EML component calculation into a constant bias!
        mask_full_const = mask_a_const & mask_c_const
        full_const_contrib = eml_w * (np.exp(eml_b) - np.log(softplus(eml_d) + self.eps))
        const_full_eml = np.sum(np.where(mask_full_const, full_const_contrib, 0.0), axis=-1)
        
        # Active masks excluding full constants
        active_mask = (np.abs(eml_w) > prune_threshold) & (~mask_full_const)
        idx_neurons, idx_components = np.where(active_mask)
        
        # Print folding stats
        total_terms = self.intermediate_size * self.num_components
        print(f"  - Total EML components: {total_terms}")
        print(f"  - Fully folded constant components: {np.sum(mask_full_const)} ({np.sum(mask_full_const)/total_terms*100:.2f}%)")
        print(f"  - Partially folded exp constants: {np.sum(mask_a_const & ~mask_full_const)} ({np.sum(mask_a_const & ~mask_full_const)/total_terms*100:.2f}%)")
        print(f"  - Partially folded log constants: {np.sum(mask_c_const & ~mask_full_const)} ({np.sum(mask_c_const & ~mask_full_const)/total_terms*100:.2f}%)")
        print(f"  - Remaining active dynamic components: {len(idx_neurons)} ({len(idx_neurons)/total_terms*100:.2f}%)")

        # Store arrays in a dictionary to pass to the closure, cast to float32
        weights_dict = {
            "w_gate_linear": w_gate_linear.astype(np.float32),
            "w_up": w_up.astype(np.float32),
            "w_down": w_down.astype(np.float32),
            "eml_a": eml_a.astype(np.float32),
            "eml_b": eml_b.astype(np.float32),
            "eml_c": eml_c.astype(np.float32),
            "eml_d": eml_d.astype(np.float32),
            "eml_w": eml_w.astype(np.float32),
            "const_exp_vals": const_exp_vals.astype(np.float32),
            "const_log_vals": const_log_vals.astype(np.float32),
            "const_full_eml": const_full_eml.astype(np.float32),
            "mask_a_const": mask_a_const,
            "mask_c_const": mask_c_const,
        }
        
        # We generate a fully vectorized NumPy DAG
        code_lines = [
            "import numpy as np",
            "",
            "def stable_softplus(x):",
            "    # Match PyTorch's softplus thresholding to prevent overflow and preserve precision",
            "    return np.where(x > 20.0, x, np.log(1.0 + np.exp(np.clip(x, -50.0, 20.0))))",
            "",
            "def eval_eml_kan_mlp_vectorized_dag(X, w):",
            "    # Input shape: [N, hidden_size]",
            "    X_f32 = X.astype(np.float32)",
            "    # 1. Base Linear Projections",
            "    gate_linear = X_f32 @ w['w_gate_linear'].T",
            "    up_proj = X_f32 @ w['w_up'].T",
            "",
            "    # 2. Sparse Vectorized EML KAN Activation with Constant Folding",
            f"    active_mask = (np.abs(w['eml_w']) > {prune_threshold}) & (~(w['mask_a_const'] & w['mask_c_const']))",
            "    idx_neurons, idx_components = np.where(active_mask)",
            "    ",
            "    # Select only active inputs and parameters",
            "    x_active = gate_linear[:, idx_neurons]",
            "    a_active = w['eml_a'][idx_neurons, idx_components]",
            "    b_active = w['eml_b'][idx_neurons, idx_components]",
            "    c_active = w['eml_c'][idx_neurons, idx_components]",
            "    d_active = w['eml_d'][idx_neurons, idx_components]",
            "    w_active = w['eml_w'][idx_neurons, idx_components]",
            "    ",
            "    # Partially folded constant lookup masks",
            "    a_const = w['mask_a_const'][idx_neurons, idx_components]",
            "    c_const = w['mask_c_const'][idx_neurons, idx_components]",
            "    ",
            "    # Precomputed constant lookups",
            "    exp_consts = w['const_exp_vals'][idx_neurons, idx_components]",
            "    log_consts = w['const_log_vals'][idx_neurons, idx_components]",
            "    ",
            "    # Compute active KAN terms dynamically, using precomputed constants where applicable",
            "    # arg_x: Use precomputed exp(b) if a_const is True",
            "    dynamic_arg_x = np.clip(a_active * x_active + b_active, -10.0, 10.0)",
            "    exp_part = np.where(a_const, exp_consts, np.exp(dynamic_arg_x))",
            "    ",
            "    # arg_y: Use precomputed log(softplus(d)) if c_const is True",
            "    dynamic_arg_y = stable_softplus(c_active * x_active + d_active) + " + f"{self.eps}",
            "    log_part = np.where(c_const, log_consts, np.log(dynamic_arg_y))",
            "    ",
            "    corr_active = w_active * (exp_part - log_part)",
            "    ",
            "    # Accumulate dynamic active corrections",
            "    eml_corr = np.zeros_like(gate_linear)",
            "    np.add.at(eml_corr, (slice(None), idx_neurons), corr_active)",
            "    ",
            "    # Add fully folded constant bias corrections",
            "    gate_out = gate_linear + eml_corr + w['const_full_eml']",
            "",
            "    # 3. GLU fusion with GELU activation",
            "    gelu_gate = 0.5 * gate_out * (1.0 + np.tanh(0.79788456 * (gate_out + 0.044715 * gate_out**3)))",
            "    activated = gelu_gate * up_proj",
            "",
            "    # 4. Down projection",
            "    out = activated @ w['w_down'].T",
            "    return out",
        ]
        
        src_code = "\n".join(code_lines)
        
        # Execute generated code to create function in local scope
        local_scope = {"np": np}
        exec(src_code, local_scope)
        eval_fn_raw = local_scope["eval_eml_kan_mlp_vectorized_dag"]
        
        # Create a closure wrapper that embeds weights_dict
        def eval_fn(X):
            return eval_fn_raw(X, weights_dict)
            
        return src_code, eval_fn


def main():
    print("="*80)
    print("           EML-KAN VECTORIZED DAG OPTIMIZER & COMPILER (CONSTANT FOLDING)")
    print("="*80)
    
    # Load PyTorch checkpoint
    weights_path = "genomics/mostoptimizedllm/llmcopyexperiement/gemma3_eml_kan/model_state_regularized.pt"
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Setup model config
    class DummyConfig:
        hidden_size = 1152
        intermediate_size = 6912
        
    config = DummyConfig()
    
    print("Instantiating PyTorch Gemma3EMLKANGatedMLP (Layer 0)...")
    pt_mlp = Gemma3EMLKANGatedMLP(config, num_components=4)
    
    # Filter state dict keys for layer 0
    layer_idx = 0
    layer_state_dict = {}
    for k, v in state_dict.items():
        if f"model.layers.{layer_idx}.mlp." in k:
            short_k = k.replace(f"model.layers.{layer_idx}.mlp.", "")
            layer_state_dict[short_k] = v
            
    pt_mlp.load_state_dict(layer_state_dict)
    pt_mlp.eval()
    
    # Compile model using our unified vectorized EML-KAN compiler with constant folding
    compiler = EMLKANUnifiedCompiler(pt_mlp, eps=1e-6)
    
    # Compile with pruning threshold 1.5e-4 and constant threshold 1e-3
    src_code, eval_dag = compiler.compile_layer(prune_threshold=1.5e-4, constant_threshold=1e-3)
    
    # Verify correctness
    print("\nVerifying mathematical correctness...")
    test_input = np.random.randn(5, config.hidden_size).astype(np.float32)
    
    # PyTorch output
    with torch.no_grad():
        pt_out = pt_mlp(torch.tensor(test_input)).numpy()
        
    # Compiled DAG output
    dag_out = eval_dag(test_input)
    
    max_diff = np.max(np.abs(pt_out - dag_out))
    mean_diff = np.mean(np.abs(pt_out - dag_out))
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    
    if max_diff < 1e-2:
        print("SUCCESS: The sparse vectorized DAG is mathematically equivalent within tolerance!")
    else:
        print("WARNING: Difference detected!")
        
    # Benchmarking
    print("\nBenchmarking speed performance...")
    num_runs = 100
    pt_input = torch.tensor(test_input)
    
    # PyTorch CPU Warmup
    for _ in range(5):
        _ = pt_mlp(pt_input)
        
    t0 = time.time()
    for _ in range(num_runs):
        _ = pt_mlp(pt_input)
    pt_time = (time.time() - t0) / num_runs * 1000.0
    
    # Compiled DAG Warmup
    for _ in range(5):
        _ = eval_dag(test_input)
        
    t0 = time.time()
    for _ in range(num_runs):
        _ = eval_dag(test_input)
    dag_time = (time.time() - t0) / num_runs * 1000.0
    
    print(f"Average Execution Latency (Input size: {test_input.shape}):")
    print(f"  PyTorch CPU (Eager):              {pt_time:.2f} ms")
    print(f"  Vectorized KAN DAG with Folding:  {dag_time:.2f} ms")
    print(f"  Speedup Factor:                   {pt_time / dag_time:.2f}x")
    print("="*80)
    
    # Save the compiled DAG code to a file for review
    compiled_file = "compiled_eml_kan_dag.py"
    with open(compiled_file, "w", encoding="utf-8") as f:
        f.write(src_code)
    print(f"Saved compiled DAG Python code to {compiled_file}")

if __name__ == "__main__":
    main()
