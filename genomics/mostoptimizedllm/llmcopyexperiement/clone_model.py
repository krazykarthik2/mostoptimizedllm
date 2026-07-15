import os
import gc
import torch
import argparse
from model import Gemma3EMLKANAttention, Gemma3EMLKANGatedMLP
from transformers import AutoConfig, AutoModelForCausalLM


def clone_to_eml_kan(model_id, save_path, device="cuda"):
    print(f"Loading original model: {model_id}...")
    original = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to(device)
    original.eval()
    config = original.config
    num_layers = config.num_hidden_layers
    print(f"Layers: {num_layers} | hidden: {config.hidden_size} | intermediate: {config.intermediate_size}")
    print(f"Attn heads: {config.num_attention_heads} | KV heads: {config.num_key_value_heads} | head_dim: {config.head_dim}")

    # ---- Build new EML-KAN state dict by copying weights layer by layer ----
    new_state = {}

    # Copy embeddings directly
    new_state["model.embed_tokens.weight"] = original.model.embed_tokens.weight.clone()
    print(f"Copied embed_tokens: {original.model.embed_tokens.weight.shape}")

    # Copy final norm
    new_state["model.norm.weight"] = original.model.norm.weight.clone()
    print(f"Copied final norm: {original.model.norm.weight.shape}")

    # Copy lm_head (tied to embeddings usually, but copy anyway)
    new_state["lm_head.weight"] = original.lm_head.weight.clone()
    print(f"Copied lm_head: {original.lm_head.weight.shape}")

    for i in range(num_layers):
        layer = original.model.layers[i]
        prefix_new = f"model.layers.{i}"
        prefix_old = f"model.layers.{i}"

        # ---- Attention: copy q/k/v/o_proj weights into EMLKANLinear ----
        # EMLKANLinear has: self.linear (nn.Linear) and self.act (EMLActivation)
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old_w = layer.self_attn._modules[proj_name].weight  # [out, in]
            new_state[f"{prefix_new}.self_attn.{proj_name}.linear.weight"] = old_w.clone()
            # EML activation params: a, b, c, d (per output channel)
            out_feat = old_w.shape[0]
            # a=0.1, b=0 -> small exp contribution
            # c=10.0, d=0 -> softplus(c*x+d) ≈ c*x for large c, so log(softplus) ≈ log(c*x)
            # This means EML path ≈ exp(0.1*x) - log(10*x) which is near-linear for moderate x
            new_state[f"{prefix_new}.self_attn.{proj_name}.act.a"] = torch.ones(out_feat) * 0.1
            new_state[f"{prefix_new}.self_attn.{proj_name}.act.b"] = torch.zeros(out_feat)
            new_state[f"{prefix_new}.self_attn.{proj_name}.act.c"] = torch.ones(out_feat) * 10.0
            new_state[f"{prefix_new}.self_attn.{proj_name}.act.d"] = torch.zeros(out_feat)

        # Copy Q/K norm (RMSNorm)
        new_state[f"{prefix_new}.self_attn.q_norm.weight"] = layer.self_attn.q_norm.weight.clone()
        new_state[f"{prefix_new}.self_attn.k_norm.weight"] = layer.self_attn.k_norm.weight.clone()

        # ---- MLP: copy gate/up/down_proj weights into EMLKANLinear / nn.Linear ----
        # gate_proj -> EMLKANLinear (has EML activation on gate)
        old_gate = layer.mlp.gate_proj.weight
        new_state[f"{prefix_new}.mlp.gate_proj.linear.weight"] = old_gate.clone()
        out_feat = old_gate.shape[0]
        new_state[f"{prefix_new}.mlp.gate_proj.act.a"] = torch.ones(out_feat) * 0.1
        new_state[f"{prefix_new}.mlp.gate_proj.act.b"] = torch.zeros(out_feat)
        new_state[f"{prefix_new}.mlp.gate_proj.act.c"] = torch.ones(out_feat) * 10.0
        new_state[f"{prefix_new}.mlp.gate_proj.act.d"] = torch.zeros(out_feat)

        # up_proj -> plain nn.Linear (no activation)
        new_state[f"{prefix_new}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight.clone()
        # down_proj -> plain nn.Linear (no activation)
        new_state[f"{prefix_new}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight.clone()

        # ---- Layer norms: copy as-is ----
        new_state[f"{prefix_new}.input_layernorm.weight"] = layer.input_layernorm.weight.clone()
        new_state[f"{prefix_new}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.clone()
        new_state[f"{prefix_new}.pre_feedforward_layernorm.weight"] = layer.pre_feedforward_layernorm.weight.clone()
        new_state[f"{prefix_new}.post_feedforward_layernorm.weight"] = layer.post_feedforward_layernorm.weight.clone()

        print(f"  Layer {i}: attn q/k/v/o + mlp gate/up/down + 4 norms copied.")
        del layer
        gc.collect()

    del original
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Verify key coverage ----
    print("\nVerifying state dict coverage...")
    # Create dummy model to get expected keys
    dummy_attn = Gemma3EMLKANAttention(config)
    dummy_mlp = Gemma3EMLKANGatedMLP(config)

    expected_keys = set()
    for i in range(num_layers):
        p = f"model.layers.{i}"
        for k in dummy_attn.state_dict():
            expected_keys.add(f"{p}.self_attn.{k}")
        for k in dummy_mlp.state_dict():
            expected_keys.add(f"{p}.mlp.{k}")
        for ln in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
            expected_keys.add(f"{p}.{ln}.weight")
    expected_keys.add("model.embed_tokens.weight")
    expected_keys.add("model.norm.weight")
    expected_keys.add("lm_head.weight")

    new_keys = set(new_state.keys())
    missing = expected_keys - new_keys
    extra = new_keys - expected_keys
    if missing:
        print(f"  WARNING: Missing {len(missing)} keys: {list(missing)[:5]}...")
    if extra:
        print(f"  WARNING: Extra {len(extra)} keys: {list(extra)[:5]}...")
    if not missing and not extra:
        print("  Perfect key match!")

    # ---- Sanity forward pass ----
    print("\nBuilding EML-KAN model for sanity check...")
    eml_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)

    # Replace attention and MLP blocks
    for i in range(num_layers):
        eml_model.model.layers[i].self_attn = Gemma3EMLKANAttention(config).to(device).to(eml_model.dtype)
        eml_model.model.layers[i].mlp = Gemma3EMLKANGatedMLP(config).to(device).to(eml_model.dtype)

    # Load the cloned state dict
    remapped = eml_model.state_dict()
    loadable = {}
    for k, v in new_state.items():
        if k in remapped and remapped[k].shape == v.shape:
            loadable[k] = v
    remapped.update(loadable)
    eml_model.load_state_dict(remapped)

    print(f"Loaded {len(loadable)}/{len(new_state)} cloned parameters.")

    # Forward pass test
    dummy_input = torch.randint(0, config.vocab_size, (1, 32), device=device)
    with torch.no_grad():
        out = eml_model(dummy_input)
    print(f"Output logits shape: {out.logits.shape}")
    print("Sanity check PASSED.")

    # ---- Save ----
    print(f"\nSaving EML-KAN cloned model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    torch.save(eml_model.state_dict(), os.path.join(save_path, "model_state.pt"))
    config.save_pretrained(save_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone Gemma-3 into EML-KAN (attention + MLP)")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--save_path", type=str, default="gemma3_eml_kan")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    clone_to_eml_kan(args.model_id, args.save_path, args.device)
