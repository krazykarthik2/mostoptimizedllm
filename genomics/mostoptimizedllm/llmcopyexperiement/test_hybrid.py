import os
import torch
from model import Gemma3EMLKANAttention, Gemma3EMLKANGatedMLP, Gemma3GatedHybridMLP
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def generate_response(model, tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            repetition_penalty=1.2, no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_hybrid_eval():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state_tuned.pt"
    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    test_prompts = [
        "I have 12 apples. I give 3 to Alice and 5 to Bob. Then I buy 7 more. How many do I have? Think step by step.",
        "Write a Python function to check if a number is prime.",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
    ]

    # ---- Test 1: Pure EML-KAN (alpha=1.0) ----
    print(f"\nLoading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)

    print("Replacing attention + MLP with EML-KAN...")
    for i in range(config.num_hidden_layers):
        model.model.layers[i].self_attn = Gemma3EMLKANAttention(config).to(device).to(model.dtype)
        model.model.layers[i].mlp = Gemma3EMLKANGatedMLP(config).to(device).to(model.dtype)

    print(f"Loading EML-KAN weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    model_sd = model.state_dict()
    loadable = {k: v for k, v in state_dict.items() if k in model_sd and model_sd[k].shape == v.shape}
    model_sd.update(loadable)
    model.load_state_dict(model_sd)
    model.eval()
    print(f"Loaded {len(loadable)} parameters.\n")

    print("=" * 60)
    print("  CONFIGURATION: Pure EML-KAN Model")
    print("=" * 60)
    for p in test_prompts:
        response = generate_response(model, tokenizer, p, device)
        print(f"\nQ: {p}")
        print(f"A: {response}")

    # ---- Test 2: Hybrid (original attention + EML-KAN MLP via gated blend) ----
    print(f"\n\nLoading fresh model for hybrid test...")
    model2 = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)

    print("Swapping MLP to Gemma3GatedHybridMLP...")
    for i in range(config.num_hidden_layers):
        original_mlp = model2.model.layers[i].mlp
        model2.model.layers[i].mlp = Gemma3GatedHybridMLP(config, original_mlp).to(model2.dtype).to(device)

    # Load EML-KAN weights into hybrid's kan_mlp sub-network
    hybrid_sd = model2.state_dict()
    for k, v in state_dict.items():
        if "mlp." in k:
            new_key = k.replace("mlp.", "mlp.kan_mlp.")
            if new_key in hybrid_sd and hybrid_sd[new_key].shape == v.shape:
                hybrid_sd[new_key] = v
    model2.load_state_dict(hybrid_sd, strict=False)
    model2.eval()

    # alpha = 1.0 -> pure EML-KAN through hybrid wrapper
    for i in range(config.num_hidden_layers):
        model2.model.layers[i].mlp.alpha.data.fill_(1.0)

    print("\n" + "=" * 60)
    print("  CONFIGURATION: Hybrid (alpha=1.0, pure EML-KAN path)")
    print("=" * 60)
    for p in test_prompts:
        response = generate_response(model2, tokenizer, p, device)
        print(f"\nQ: {p}")
        print(f"A: {response}")

    # alpha = 0.0 -> original MLP
    for i in range(config.num_hidden_layers):
        model2.model.layers[i].mlp.alpha.data.fill_(0.0)

    print("\n" + "=" * 60)
    print("  CONFIGURATION: Hybrid (alpha=0.0, original MLP baseline)")
    print("=" * 60)
    for p in test_prompts:
        response = generate_response(model2, tokenizer, p, device)
        print(f"\nQ: {p}")
        print(f"A: {response}")


if __name__ == "__main__":
    run_hybrid_eval()
