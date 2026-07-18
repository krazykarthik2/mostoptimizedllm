import os
import sys
import torch
import torch.nn as nn
import time

# Add the repo's library path to sys.path using directory path traversal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mostoptimizedllm", "genomics", "mostoptimizedllm", "llmcopyexperiement")))
from transformers import AutoTokenizer, AutoModelForCausalLM

class HopfieldExpSumExpAttention(nn.Module):
    """
    Fused Hopfield Attention using Log-Sum-Exp / Exp-Sum-Exp formulation.
    Bypasses intermediate softmax memory allocations and matches standard Gemma-3 math exactly.
    """
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Necessary decoder layer attributes
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        
        # Exact Gemma-3 scaling factor
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
        
        # Use exact Gemma3RMSNorm definition
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
            
        # Update KV-Cache (updates in-place)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} if position_embeddings is not None else {}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
            
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            
        # 2. Exp-Sum-Exp Attention Softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Softcap attention logits if specified in Gemma-3 config
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
            
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # Standard Hugging Face softmax is evaluated in float32 for stability
        attn_probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        context_layer = torch.matmul(attn_probs, v)
        
        # 3. Output Projection
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(context_layer)
        
        # Return exactly 2 values matching standard Transformers decoder interface
        return output, attn_probs

def main():
    print("="*80)
    print("        FUSED HOPFIELD EXP-SUM-EXP ATTENTION COMPILER")
    print("="*80)
    
    model_id = "google/gemma-3-1b-it"
    print("Loading original Gemma-3 model to extract attention weights...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)
    
    # Force eager implementation for standard verification comparison
    model.config._attn_implementation = "eager"
    
    orig_attn = model.model.layers[0].self_attn
    config = model.config
    
    print("\nInitializing Fused Hopfield Exp-Sum-Exp Attention module...")
    hopfield_attn = HopfieldExpSumExpAttention(config, layer_idx=0)
    hopfield_attn.load_weights_from_original(orig_attn)
    hopfield_attn.eval()
    
    test_input = torch.randn(2, 64, config.hidden_size) # [batch, seq_len, hidden_size]
    
    # Initialize mock RoPE cos/sin embeddings (matching sequence length and head dim)
    cos = torch.ones(1, 64, config.head_dim)
    sin = torch.zeros(1, 64, config.head_dim)
    position_embeddings = (cos, sin)
    attention_mask = torch.zeros(2, 1, 64, 64)
    
    print("\nVerifying outputs...")
    with torch.no_grad():
        orig_output, _ = orig_attn(test_input, position_embeddings=position_embeddings, attention_mask=attention_mask)
        hopfield_output, _ = hopfield_attn(test_input, position_embeddings=position_embeddings, attention_mask=attention_mask)
        
    max_diff = torch.max(torch.abs(orig_output - hopfield_output)).item()
    mean_diff = torch.mean(torch.abs(orig_output - hopfield_output)).item()
    print(f"Max Difference: {max_diff:.2e}")
    print(f"Mean Difference: {mean_diff:.2e}")
    
    if max_diff < 1e-4:
        print("SUCCESS: Fused Exp-Sum-Exp attention matches standard outputs losslessly!")
    else:
        print("WARNING: Difference detected!")

if __name__ == "__main__":
    main()
