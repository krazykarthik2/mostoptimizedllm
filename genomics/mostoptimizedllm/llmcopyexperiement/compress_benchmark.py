import os
import gc
import time
import torch
import torch.nn as nn
import argparse
from model import Gemma3EMLKANAttention, Gemma3EMLKANGatedMLP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TCS_NQT_BENCHMARKS = [
    {
        "category": "Arithmetic Patterns",
        "prompt": "Find the 15th term of the arithmetic progression: 3, 7, 11, 15, ... Show step-by-step reasoning and state the final answer.",
    },
    {
        "category": "Logical Sufficiency",
        "prompt": "In a family, A is the brother of B, B is the sister of C, and C is the father of D. How is A related to D? Options: (a) Uncle, (b) Brother, (c) Grandfather, (d) Cousin. Choose the correct option and justify.",
    },
    {
        "category": "Arithmetic Aptitude",
        "prompt": "A shopkeeper sells a book at a 20% discount on the marked price and still earns a profit of 12%. What is the ratio of cost price to marked price? Explain your calculation.",
    },
    {
        "category": "Reasoning & Coding",
        "prompt": "What is the output of a function that returns the first non-repeating character in the string 'statistics'? Show the step-by-step trace.",
    },
    {
        "category": "Instruction Following",
        "prompt": "Generate a JSON object with keys: 'name', 'roll_no', 'subjects', 'grade'. Output only valid JSON.",
    },
    {
        "category": "Long Horizon Reasoning",
        "prompt": "A farmer has 100 meters of fencing. He wants to enclose a rectangular garden along a river (no fence needed on the river side). What dimensions maximize the area? Show all reasoning steps.",
    },
    {
        "category": "Multi-Step Math",
        "prompt": "If x + y = 10 and x - y = 4, find x^2 + y^2. Show your work step by step.",
    },
]


def apply_magnitude_pruning(model, sparsity=0.5):
    print(f"\n[PRUNING] Applying magnitude pruning to {sparsity*100:.0f}% sparsity...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and ("linear.weight" in name or "weight_eml" in name):
                threshold = torch.quantile(torch.abs(param.float()), sparsity)
                mask = torch.abs(param) >= threshold
                param.mul_(mask.to(param.dtype))
                zeros = (param == 0).sum().item()
                total = param.numel()
                print(f"  {name}: {zeros/total*100:.1f}% sparse ({zeros}/{total})")


def evaluate_model(model, tokenizer, benchmarks, device="cuda"):
    model.eval()
    results = []
    print("\nRunning downstream benchmarks...")
    for item in benchmarks:
        print(f"\n--- {item['category']} ---")
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=False,
                repetition_penalty=1.2, no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )
        latency = time.time() - t0
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Response ({latency:.2f}s): {response.strip()[:200]}")
        results.append({"category": item["category"], "response": response.strip(), "latency": latency})
    return results


def run_compression_and_benchmark(model_id, weights_path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    print("\n--- LOADING DISTILLED EML-KAN MODEL ---")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    for i in range(config.num_hidden_layers):
        model.model.layers[i].self_attn = Gemma3EMLKANAttention(config).to(device).to(model.dtype)
        model.model.layers[i].mlp = Gemma3EMLKANGatedMLP(config).to(device).to(model.dtype)

    state_dict = torch.load(weights_path, map_location=device)
    model_sd = model.state_dict()
    loadable = {k: v for k, v in state_dict.items() if k in model_sd and model_sd[k].shape == v.shape}
    model_sd.update(loadable)
    model.load_state_dict(model_sd)
    print(f"Loaded {len(loadable)} parameters.")

    print("\n[EVAL] Evaluating Distilled EML-KAN Model (Baseline)...")
    kan_results = evaluate_model(model, tokenizer, TCS_NQT_BENCHMARKS, device=device)

    apply_magnitude_pruning(model, sparsity=0.5)

    print("\n[QUANTIZATION] Transferring to CPU for 8-bit Dynamic Quantization...")
    model_cpu = model.cpu().float()
    quantized_model = torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    print("Dynamic quantization applied!")

    temp_dir = "temp_compression_metrics"
    os.makedirs(temp_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(temp_dir, "uncompressed.pt"))
    torch.save(quantized_model.state_dict(), os.path.join(temp_dir, "quantized.pt"))
    u_size = os.path.getsize(os.path.join(temp_dir, "uncompressed.pt")) / (1024*1024)
    q_size = os.path.getsize(os.path.join(temp_dir, "quantized.pt")) / (1024*1024)
    print(f"  Uncompressed: {u_size:.2f} MB | Quantized: {q_size:.2f} MB | Ratio: {u_size/q_size:.2f}x")
    os.remove(os.path.join(temp_dir, "uncompressed.pt"))
    os.remove(os.path.join(temp_dir, "quantized.pt"))
    os.rmdir(temp_dir)

    gc.collect()
    torch.cuda.empty_cache()

    print("\n[EVAL] Evaluating Compressed (Pruned + Quantized) Model on CPU...")
    quantized_results = evaluate_model(quantized_model, tokenizer, TCS_NQT_BENCHMARKS, device="cpu")

    print("\n" + "=" * 80)
    print("                      COMPARATIVE EVALUATION SUMMARY")
    print("=" * 80)
    for idx, item in enumerate(TCS_NQT_BENCHMARKS):
        print(f"Category: {item['category']}")
        print(f"  EML-KAN Latency: {kan_results[idx]['latency']:.2f}s | Quantized (CPU): {quantized_results[idx]['latency']:.2f}s")
        print(f"  EML-KAN: {kan_results[idx]['response'][:120]}...")
        print(f"  Quantized: {quantized_results[idx]['response'][:120]}...")
        print("-" * 60)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase C: Edge Compression & Benchmarking")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--weights_path", type=str, default="gemma3_eml_kan/model_state_tuned.pt")
    args = parser.parse_args()
    run_compression_and_benchmark(args.model_id, args.weights_path)
