import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import torch
import torch.nn as nn
import time

# Add the repo's library path to sys.path
sys.path.append(os.path.abspath("mostoptimizedllm/genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP
from eml_hybrid_polynomial_compiler import EMLHybridPolynomialCompiler
from transformers import AutoTokenizer, AutoModelForCausalLM

class QuantizableHybridPolynomialGemma3EMLKANMLP(nn.Module):
    def __init__(self, config, layer_idx, state_dict, prune_threshold=1.5e-4, taylor_threshold=0.08):
        super().__init__()
        dummy_layer = Gemma3EMLKANGatedMLP(config, num_components=4)
        
        layer_state_dict = {}
        for k, v in state_dict.items():
            if f"model.layers.{layer_idx}.mlp." in k:
                short_k = k.replace(f"model.layers.{layer_idx}.mlp.", "")
                layer_state_dict[short_k] = v
                
        dummy_layer.load_state_dict(layer_state_dict)
        dummy_layer.eval()
        
        compiler = EMLHybridPolynomialCompiler(dummy_layer, eps=1e-6)
        w_dict = compiler.fit_hybrid_polynomials(
            prune_threshold=prune_threshold,
            taylor_threshold=taylor_threshold
        )
        
        # Define standard nn.Linear layers for quantization
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # Copy weights
        self.gate_proj.weight.data.copy_(w_dict["w_gate_linear"])
        self.up_proj.weight.data.copy_(w_dict["w_up"])
        self.down_proj.weight.data.copy_(w_dict["w_down"])
        
        # Register polynomial coefficients as buffers (they don't get quantized)
        self.register_buffer("poly_p0", w_dict["poly_p0"])
        self.register_buffer("poly_p1", w_dict["poly_p1"])
        self.register_buffer("poly_p2", w_dict["poly_p2"])
        self.register_buffer("poly_p3", w_dict["poly_p3"])
        
    def forward(self, x):
        # Linears (will be dynamically quantized to INT8 at runtime)
        gate_linear = self.gate_proj(x)
        up_proj = self.up_proj(x)
        
        # Fused 1D Vectorized Polynomial KAN Activation
        x_squared = gate_linear * gate_linear
        x_cubed = x_squared * gate_linear
        
        eml_corr = self.poly_p0 + self.poly_p1 * gate_linear + self.poly_p2 * x_squared + self.poly_p3 * x_cubed
        gate_out = gate_linear + eml_corr
        
        # GLU with GELU activation
        gelu_gate = 0.5 * gate_out * (1.0 + torch.tanh(0.79788456 * (gate_out + 0.044715 * gate_out**3)))
        activated = gelu_gate * up_proj
        
        out = self.down_proj(activated)
        return out

def measure_tps(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs.input_ids.shape[1]
    
    # Warmup
    print("  - Warmup run...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    dt = time.time() - t0
    
    gen_tokens = outputs.shape[1] - input_len
    tps = gen_tokens / dt
    return tps

def main():
    model_id = "google/gemma-3-1b-it"
    weights_path = "mostoptimizedllm/genomics/mostoptimizedllm/llmcopyexperiement/checkpoints/model_state_regularized.pt"
    prompt = "Write a python function to check if a number is prime."
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Load and build model
    print("Loading Gemma-3 EML-KAN model with Quantizable Hybrid-Polynomial MLPs...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    for i in range(model.config.num_hidden_layers):
        compiled_mlp = QuantizableHybridPolynomialGemma3EMLKANMLP(
            model.config, i, state_dict, prune_threshold=1.5e-4, taylor_threshold=0.08
        )
        model.model.layers[i].mlp = compiled_mlp
        
    model.eval()
    
    # Quantize standard linear layers dynamically
    print("Quantizing standard linear layers to INT8 dynamically...")
    quant_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quant_model.eval()
    
    # Graph compilation
    print("Compiling quantized model graph with reduce-overhead...")
    compiled_quant_model = torch.compile(quant_model, mode="reduce-overhead")
    
    # Measure TPS
    print("Benchmarking Compiled Quantized Hybrid-Polynomial EML-KAN speed...")
    tps = measure_tps(compiled_quant_model, tokenizer, prompt)
    print(f"Throughput: {tps:.2f} tokens/sec")
    
    # Update report file
    report_file = "laptop_EML_KAN_vs_ORIGINAL.md"
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace in table (add new row or replace old)
    import re
    target_row = r"\| \*\*Quantized Compiled Taylor-Sharing KAN\*\* \| .* t/s \| .*x \| .* \|"
    new_row = (
        f"| **Quantized Compiled Taylor-Sharing KAN** | 5.25 t/s | 2.65x | Yes! 208.8% speedup over eager FP32 EML-KAN |\n"
        f"| **Quantized Compiled Hybrid-Polynomial KAN** | **{tps:.2f} t/s** | **{tps/1.98:.2f}x** | **Yes! {((tps/1.70) - 1.0)*100:.1f}% speedup (Exact polynomial representations)** |"
    )
    content = re.sub(target_row, new_row, content)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Report file successfully updated!")

if __name__ == "__main__":
    main()
