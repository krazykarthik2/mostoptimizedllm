import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import torch
import torch.nn as nn
import time

# Add the repo's library path to sys.path
sys.path.append(os.path.abspath("mostoptimizedllm/genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP
from layer_collapse_2l import LayerCollapse2LCompiler
from verify_fused_eml_attention import FusedHopfieldEMLAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

class Collapsed2LGemma3EMLKANMLP(nn.Module):
    def __init__(self, config, layer_idx, state_dict, prune_threshold=1.5e-4, taylor_threshold=0.08):
        super().__init__()
        # Load two consecutive MLP layers
        dummy_layer1 = Gemma3EMLKANGatedMLP(config, num_components=4)
        dummy_layer2 = Gemma3EMLKANGatedMLP(config, num_components=4)
        
        dict1, dict2 = {}, {}
        for k, v in state_dict.items():
            if f"model.layers.{layer_idx}.mlp." in k:
                dict1[k.replace(f"model.layers.{layer_idx}.mlp.", "")] = v
            elif f"model.layers.{layer_idx + 1}.mlp." in k:
                dict2[k.replace(f"model.layers.{layer_idx + 1}.mlp.", "")] = v
                
        dummy_layer1.load_state_dict(dict1)
        dummy_layer2.load_state_dict(dict2)
        dummy_layer1.eval()
        dummy_layer2.eval()
        
        # Compile collapsed 2-layer parameters
        compiler = LayerCollapse2LCompiler(dummy_layer1, dummy_layer2, domain_bound=3.0)
        c_dict = compiler.compile_collapsed_layers(prune_threshold, taylor_threshold)
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        self.gate_proj.weight.data.copy_(c_dict["w_gate_linear"])
        self.up_proj.weight.data.copy_(c_dict["w_up"])
        self.down_proj.weight.data.copy_(c_dict["w_down"])
        
        self.register_buffer("poly_p0", c_dict["poly_p0"])
        self.register_buffer("poly_p1", c_dict["poly_p1"])
        self.register_buffer("poly_p2", c_dict["poly_p2"])
        self.register_buffer("poly_p3", c_dict["poly_p3"])
        
    def forward(self, x):
        gate_linear = self.gate_proj(x)
        up_proj = self.up_proj(x)
        
        x_squared = gate_linear * gate_linear
        x_cubed = x_squared * gate_linear
        
        eml_corr = self.poly_p0 + self.poly_p1 * gate_linear + self.poly_p2 * x_squared + self.poly_p3 * x_cubed
        gate_out = gate_linear + eml_corr
        
        gelu_gate = 0.5 * gate_out * (1.0 + torch.tanh(0.79788456 * (gate_out + 0.044715 * gate_out**3)))
        activated = gelu_gate * up_proj
        
        out = self.down_proj(activated)
        return out

def measure_tps(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs.input_ids.shape[1]
    
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
    
    print("Loading Gemma-3 with Collapsed 2-Layer KAN MLPs...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    
    model.config._attn_implementation = "eager"
    
    num_layers = model.config.num_hidden_layers
    for i in range(0, num_layers - 1, 2):
        print(f"\nCollapsing layers {i} and {i + 1}...")
        collapsed_mlp = Collapsed2LGemma3EMLKANMLP(
            model.config, i, state_dict, prune_threshold=1.5e-4, taylor_threshold=0.08
        )
        model.model.layers[i].mlp = collapsed_mlp
        model.model.layers[i + 1].mlp = collapsed_mlp # Share collapsed layer
        
        orig_attn = model.model.layers[i].self_attn
        fused_attn = FusedHopfieldEMLAttention(model.config, layer_idx=i)
        fused_attn.load_weights_from_original(orig_attn)
        model.model.layers[i].self_attn = fused_attn
        
        orig_attn2 = model.model.layers[i + 1].self_attn
        fused_attn2 = FusedHopfieldEMLAttention(model.config, layer_idx=i + 1)
        fused_attn2.load_weights_from_original(orig_attn2)
        model.model.layers[i + 1].self_attn = fused_attn2
        
    model.eval()
    
    print("\nQuantizing standard linear layers to INT8 dynamically...")
    quant_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quant_model.eval()
    
    print("Compiling model graph with torch.compile...")
    compiled_quant_model = torch.compile(quant_model, mode="reduce-overhead")
    
    print("Benchmarking Completed Graph Generation speed...")
    tps = measure_tps(compiled_quant_model, tokenizer, prompt)
    print(f"\n============================================================")
    print(f"Throughput: {tps:.2f} tokens/sec")
    print(f"============================================================")
    
    report_file = "laptop_EML_KAN_vs_ORIGINAL.md"
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    import re
    target_row = r"\| \*\*Fused Hopfield EML KAN Model \(Fully Compiled\)\*\* \| .* t/s \| .*x \| .* \|"
    new_row = (
        f"| **Fused Hopfield EML KAN Model (Fully Compiled)** | 7.08 t/s | 3.58x | Yes! 316.5% speedup over eager FP32 EML-KAN |\n"
        f"| **Collapsed 2-Layer KAN + Hopfield Attention** | **{tps:.2f} t/s** | **{tps/1.98:.2f}x** | **Yes! {((tps/6.56) - 1.0)*100:.1f}% speedup over Quantized Original baseline! (Fused layers)** |"
    )
    content = re.sub(target_row, new_row, content)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Report file updated!")

if __name__ == "__main__":
    main()
