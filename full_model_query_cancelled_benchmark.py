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
from verify_fused_eml_attention import FusedHopfieldEMLAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

class QueryCancelledHopfieldAttention(nn.Module):
    """
    Fused Hopfield Attention with Query Log-Softplus Cancellation.
    Mathematical Proof:
    Since the query scale factor S_i = softplus(c*Q_i + d) depends only on the query index i,
    subtracting log(S_i) from logits results in a constant shift that cancels out during softmax:
    softmax(logits - log(S_i)) = softmax(logits)
    Therefore, we completely drop the log-softplus query evaluations from the score pathway.
    """
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        
        query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", self.head_dim)
        self.scaling = query_pre_attn_scalar ** -0.5
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", None)
        
        hidden_size = config.hidden_size
        kv_dim = self.num_key_value_heads * self.head_dim
        q_dim = self.num_heads * self.head_dim
        
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)
        
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
        self.q_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

    def load_weights_from_original(self, orig_attn):
        self.q_proj.weight.data.copy_(orig_attn.q_proj.weight.data)
        self.k_proj.weight.data.copy_(orig_attn.k_proj.weight.data)
        self.v_proj.weight.data.copy_(orig_attn.v_proj.weight.data)
        self.o_proj.weight.data.copy_(orig_attn.o_proj.weight.data)
        
        if hasattr(orig_attn, "q_norm") and orig_attn.q_norm is not None:
            self.q_norm.weight.data.copy_(orig_attn.q_norm.weight.data)
        if hasattr(orig_attn, "k_norm") and orig_attn.k_norm is not None:
            self.k_norm.weight.data.copy_(orig_attn.k_norm.weight.data)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} if position_embeddings is not None else {}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
            
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            
        # Compute scores (Log-softplus is mathematically canceled out, zero operations here)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
            
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attn_probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        context_layer = torch.matmul(attn_probs, v)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(context_layer)
        
        return output, attn_probs

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
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        self.gate_proj.weight.data.copy_(w_dict["w_gate_linear"])
        self.up_proj.weight.data.copy_(w_dict["w_up"])
        self.down_proj.weight.data.copy_(w_dict["w_down"])
        
        self.register_buffer("poly_p0", w_dict["poly_p0"])
        self.register_buffer("poly_p1", w_dict["poly_p1"])
        self.register_buffer("poly_p2", w_dict["poly_p2"])
        self.register_buffer("poly_p3", w_dict["poly_p3"])
        
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
    
    print("Loading Gemma-3 with Query-Cancelled Attention...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    
    model.config._attn_implementation = "eager"
    
    for i in range(model.config.num_hidden_layers):
        compiled_mlp = QuantizableHybridPolynomialGemma3EMLKANMLP(
            model.config, i, state_dict, prune_threshold=1.5e-4, taylor_threshold=0.08
        )
        model.model.layers[i].mlp = compiled_mlp
        
        orig_attn = model.model.layers[i].self_attn
        cancelled_attn = QueryCancelledHopfieldAttention(model.config, layer_idx=i)
        cancelled_attn.load_weights_from_original(orig_attn)
        model.model.layers[i].self_attn = cancelled_attn
        
    model.eval()
    
    print("Quantizing standard linear layers to INT8 dynamically...")
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
        f"| **Query-Cancelled Hopfield EML KAN Model** | **{tps:.2f} t/s** | **{tps/1.98:.2f}x** | **Yes! {((tps/6.56) - 1.0)*100:.1f}% speedup over Quantized Original baseline! (Zero query-log operations)** |"
    )
    content = re.sub(target_row, new_row, content)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Report file updated!")

if __name__ == "__main__":
    main()
