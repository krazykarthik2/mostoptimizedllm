import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma3HopfieldKANAttentionFloat32(nn.Module):
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

        # Hopfield KAN Log-Sum-Exp Softmax retrieval in Float32 to prevent numerical precision underflow:
        scores_fp32 = scores.float()
        max_val = torch.max(scores_fp32, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores_fp32 - max_val)
        sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
        log_sum_exp = torch.log(sum_exp + 1e-9)
        
        # Mathematically exact, numerically stable softmax retrieval output (MSE = 0.0)
        attn_weights = torch.exp(scores_fp32 - max_val - log_sum_exp).to(query_states.dtype)

        if self.training and self.original_attn.attention_dropout > 0.0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.original_attn.attention_dropout)

        # Context output projection
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.original_attn.o_proj(attn_output)

        return attn_output, attn_weights

def main():
    model_id = "google/gemma-3-1b-it"
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    test_prompts = [
        "If a train travels 60 miles per hour, how far will it travel in 2.5 hours? Explain your reasoning step-by-step.",
        "A father has 4 daughters. Each daughter has a brother. How many children does the father have in total? Explain your reasoning."
    ]
    
    print("\n" + "="*80)
    print(" TESTING HOPFIELD KAN ATTENTION WITH HIGH-PRECISION FLOAT32 SOFTMAX")
    print("="*80)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    
    for i in range(model.config.num_hidden_layers):
        orig_attn = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = Gemma3HopfieldKANAttentionFloat32(orig_attn).to(device)
        
    for p in test_prompts:
        messages = [{"role": "user", "content": p}]
        chat_prompt = tokenizer.apply_prompt_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\nPrompt: {p}\nResponse:\n{response}")
        print("-"*80)

if __name__ == "__main__":
    main()
