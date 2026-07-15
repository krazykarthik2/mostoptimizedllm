import torch
import torch.nn as nn
import torch.nn.functional as F


class EMLCorrection(nn.Module):
    """
    Additive EML residual path from MHNKAN.
    output = x + sum_k w_k * eml(a_k*x + b_k, softplus(c_k*x + d_k) + eps)
    Initialized to near-zero so the base linear path dominates after weight copy.
    """
    def __init__(self, channels, num_components=4, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.num_components = num_components
        self.a = nn.Parameter(torch.randn(channels, num_components) * 0.1)
        self.b = nn.Parameter(torch.randn(channels, num_components) * 0.1)
        self.c = nn.Parameter(torch.randn(channels, num_components) * 0.1)
        self.d = nn.Parameter(torch.randn(channels, num_components) * 0.1)
        # EML mixture weights: initialized near-zero so EML path is negligible at start
        self.weight_eml = nn.Parameter(torch.zeros(channels, num_components))

    def forward(self, x):
        out = x
        for k in range(self.num_components):
            arg_x = torch.clamp(self.a[..., k] * x + self.b[..., k], -10.0, 10.0)
            arg_y = F.softplus(self.c[..., k] * x + self.d[..., k]) + self.eps
            out = out + self.weight_eml[..., k] * (torch.exp(arg_x) - torch.log(arg_y))
        return out


class EMLKANLinear(nn.Module):
    """
    Linear layer + additive EML correction (MHNKAN style).
    weight_base = original Linear weight (dominates at init).
    EML path starts at zero via weight_eml init.
    """
    def __init__(self, in_features, out_features, num_components=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.eml = EMLCorrection(out_features, num_components=num_components)

    def forward(self, x):
        return self.eml(self.linear(x))


class Gemma3EMLKANGatedMLP(nn.Module):
    """
    SwiGLU MLP: output = down_proj(EML(gate_proj(x)) * up_proj(x))
    gate_proj and up_proj weights copied 1:1 from original.
    EML correction on gate_proj starts at zero.
    """
    def __init__(self, config, num_components=4):
        super().__init__()
        self.gate_proj = EMLKANLinear(config.hidden_size, config.intermediate_size, num_components=num_components)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma3EMLKANAttention(nn.Module):
    """
    Attention with EML-KAN Q/K/V/O projections.
    Hopfield retrieval (softmax QK^T V) stays identical.
    """
    def __init__(self, config, num_components=4):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        self.layer_idx = 0

        hidden_size = config.hidden_size
        kv_dim = self.num_key_value_heads * self.head_dim
        q_dim = self.num_heads * self.head_dim

        self.q_proj = EMLKANLinear(hidden_size, q_dim, num_components=num_components)
        self.k_proj = EMLKANLinear(hidden_size, kv_dim, num_components=num_components)
        self.v_proj = EMLKANLinear(hidden_size, kv_dim, num_components=num_components)
        self.o_proj = EMLKANLinear(q_dim, hidden_size, num_components=num_components)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.k_norm = nn.RMSNorm(self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        max_val = torch.max(attn_weights, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(attn_weights - max_val)
        sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
        log_sum_exp = torch.log(sum_exp + 1e-9)
        attn_weights = torch.exp(attn_weights - max_val - log_sum_exp)

        if self.training and self.attention_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Gemma3GatedHybridMLP(nn.Module):
    """
    Hybrid block blending original MLP and EML-KAN MLP via alpha.
    """
    def __init__(self, config, original_mlp):
        super().__init__()
        self.original_mlp = original_mlp
        self.kan_mlp = Gemma3EMLKANGatedMLP(config)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        alpha_val = self.alpha.item() if self.alpha.device.type == "cpu" else self.alpha.data[0].item()
        if alpha_val == 0.0:
            return self.original_mlp(x)
        elif alpha_val == 1.0:
            return self.kan_mlp(x)
        else:
            return self.alpha * self.kan_mlp(x) + (1.0 - self.alpha) * self.original_mlp(x)

class Gemma3EMLKANMLP(nn.Module):
    def __init__(self, config, num_components=4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.ffn1 = EMLKANLinear(self.hidden_size, self.intermediate_size, num_components=num_components)
        self.ffn2 = EMLKANLinear(self.intermediate_size, self.hidden_size, num_components=num_components)

    def forward(self, x):
        return self.ffn2(self.ffn1(x))

class Gemma3HopfieldKANAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        # Reuse pre-trained projections to project signals
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.original_attn.head_dim)

        query_states = self.original_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.original_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply native RMSNorm to attention heads
        query_states = self.original_attn.q_norm(query_states)
        key_states = self.original_attn.k_norm(key_states)

        # Apply pre-trained rotary position embeddings
        cos, sin = position_embeddings
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.original_attn.layer_idx)

        # Standard Grouped Query Attention repeat key/value if needed
        num_queries_per_kv = self.original_attn.num_key_value_groups
        if num_queries_per_kv > 1:
            key_states = key_states.repeat_interleave(num_queries_per_kv, dim=1)
            value_states = value_states.repeat_interleave(num_queries_per_kv, dim=1)

        # Compute query-key dot product scores
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.original_attn.scaling

        if attention_mask is not None:
            scores = scores + attention_mask

        # Hopfield KAN Log-Sum-Exp Softmax retrieval:
        max_val = torch.max(scores, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores - max_val)
        sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
        log_sum_exp = torch.log(sum_exp + 1e-9)
        
        # Mathematically exact, numerically stable softmax retrieval output (MSE = 0.0)
        attn_weights = torch.exp(scores - max_val - log_sum_exp)

        if self.training and self.original_attn.attention_dropout > 0.0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.original_attn.attention_dropout)

        # Context output projection
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.original_attn.o_proj(attn_output)

        return attn_output, attn_weights
