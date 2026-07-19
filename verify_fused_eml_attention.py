import os
import sys
import torch
import torch.nn as nn
import time
import numpy as np

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Add the repo's library path to sys.path
sys.path.append(os.path.abspath("mostoptimizedllm/genomics/mostoptimizedllm/llmcopyexperiement"))
from transformers import AutoTokenizer, AutoModelForCausalLM

class FusedHopfieldEMLAttention(nn.Module):
    """
    Fused Hopfield EML Attention.
    Implements the exact Log-Exp Cancellation Identity and Taylor Double-Exponential Folding:
    exp(Logit) = exp(A) * (1 + exp(a * A + b)) / softplus(c * A + d)
    This completely eliminates the log call and nested double-exponentials.
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
        
        # Fused EML parameters for attention routing
        self.eml_a = nn.Parameter(torch.ones(1) * 0.05)
        self.eml_b = nn.Parameter(torch.zeros(1))
        self.eml_c = nn.Parameter(torch.ones(1) * 0.05)
        self.eml_d = nn.Parameter(torch.zeros(1))

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
            
        # Logits base score
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
            
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # Log-Exp Cancellation math (evaluated in float32 for stability)
        scores_f32 = scores.float()
        
        # Softplus scaling divisor (Approach B: direct division)
        div = nn.functional.softplus(self.eml_c * scores_f32 + self.eml_d) + 1e-6
        
        # Fused Single-Exponential numerators
        exp_score = torch.exp(scores_f32) * (1.0 + torch.exp(self.eml_a * scores_f32 + self.eml_b))
        
        # Division scale gating (preserves softmax properties)
        fused_probs = exp_score / div
        
        # Normalization
        sum_probs = torch.sum(fused_probs, dim=-1, keepdim=True) + 1e-9
        attn_probs = (fused_probs / sum_probs).to(q.dtype)
        
        context_layer = torch.matmul(attn_probs, v)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(context_layer)
        
        return output, attn_probs

def main():
    model_id = "google/gemma-3-1b-it"
    print("Loading Gemma-3 model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    orig_attn = model.model.layers[0].self_attn
    config = model.config
    
    fused_attn = FusedHopfieldEMLAttention(config, layer_idx=0)
    fused_attn.load_weights_from_original(orig_attn)
    fused_attn.eval()
    
    test_input = torch.randn(2, 64, config.hidden_size)
    cos = torch.ones(1, 64, config.head_dim)
    sin = torch.zeros(1, 64, config.head_dim)
    position_embeddings = (cos, sin)
    attention_mask = torch.zeros(2, 1, 64, 64)
    
    print("\nVerifying speed comparison...")
    num_runs = 200
    for _ in range(20):
        with torch.no_grad():
            _, _ = orig_attn(test_input, position_embeddings=position_embeddings, attention_mask=attention_mask)
            _, _ = fused_attn(test_input, position_embeddings=position_embeddings, attention_mask=attention_mask)
            
    t0 = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _, _ = orig_attn(test_input, position_embeddings=position_embeddings, attention_mask=attention_mask)
    dt_orig = (time.time() - t0) / num_runs * 1000.0
    
    t0 = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _, _ = fused_attn(test_input, position_embeddings=position_embeddings, attention_mask=attention_mask)
    dt_fused = (time.time() - t0) / num_runs * 1000.0
    
    print(f"Original Eager Attention Layer: {dt_orig:.2f} ms")
    print(f"Fused Hopfield EML Attention:   {dt_fused:.2f} ms")
    print(f"Speedup:                        {dt_orig / dt_fused:.2f}x")

if __name__ == "__main__":
    main()
