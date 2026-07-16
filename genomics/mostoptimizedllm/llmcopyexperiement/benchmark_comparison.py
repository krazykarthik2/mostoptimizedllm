import os
import time
import torch
import torch.nn as nn
from model import Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_performance_test(model, tokenizer, test_prompts, config_name, device):
    print(f"\nRunning performance benchmark for configuration: {config_name}...")
    
    ttfts = []
    throughputs = []
    latencies = []
    total_tokens_gen = 0
    
    # Warm-up (especially crucial for compiled models)
    warmup_inputs = tokenizer("Warm up", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
        
    for p in test_prompts:
        messages = [{"role": "user", "content": p}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        
        # TTFT
        t_start = time.time()
        with torch.no_grad():
            outputs_first = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        t_first = time.time()
        ttft = (t_first - t_start) * 1000.0  # ms
        ttfts.append(ttft)
        
        # Throughput
        t_gen_start = time.time()
        with torch.no_grad():
            outputs_full = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        t_gen_end = time.time()
        
        latency = t_gen_end - t_gen_start
        latencies.append(latency)
        
        tokens_generated = outputs_full[0].shape[0] - input_len
        total_tokens_gen += tokens_generated
        
        throughput = tokens_generated / latency if latency > 0 else 0
        throughputs.append(throughput)
        
    avg_ttft = sum(ttfts) / len(ttfts)
    avg_throughput = sum(throughputs) / len(throughputs)
    avg_latency = sum(latencies) / len(latencies)
    
    return {
        "avg_ttft": avg_ttft,
        "avg_throughput": avg_throughput,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens_gen
    }

def main():
    print("="*80)
    print("        GEMMA-3 VS EML-KAN HYBRID: DEEP PERFORMANCE COMPARISON")
    print("="*80)
    
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state_regularized.pt"
    device = "cuda:0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    test_prompts = [
        "If a train travels 60 miles per hour, how far will it travel in 2.5 hours? Explain your reasoning step-by-step.",
        "Write a python function to check if a given integer is a prime number."
    ]
    
    results = {}
    
    # --------------------------------------------------------------------------
    # Configuration 1: Original Model (Uncompiled)
    # --------------------------------------------------------------------------
    print("Loading Original Gemma-3 model...")
    original_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    original_model.eval()
    
    results["Original (Uncompiled)"] = run_performance_test(
        original_model, tokenizer, test_prompts, "Original (Uncompiled)", device
    )
    
    # --------------------------------------------------------------------------
    # Configuration 2: Original Model (Compiled)
    # --------------------------------------------------------------------------
    print("\nCompiling Original Gemma-3 model (torch.compile)...")
    t_comp_start = time.time()
    compiled_original = torch.compile(original_model)
    # Trigger compilation
    warmup_inputs = tokenizer("Compile warm up", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = compiled_original.generate(**warmup_inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    print(f"Original compilation time: {time.time() - t_comp_start:.2f} seconds.")
    
    results["Original (Compiled)"] = run_performance_test(
        compiled_original, tokenizer, test_prompts, "Original (Compiled)", device
    )
    
    del original_model, compiled_original
    torch.cuda.empty_cache()
    
    # --------------------------------------------------------------------------
    # Configuration 3: EML-KAN Model (Uncompiled)
    # --------------------------------------------------------------------------
    print("\nLoading EML-KAN Hybrid model...")
    kan_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    
    # Swap layers
    for i in range(kan_model.config.num_hidden_layers):
        kan_mlp = Gemma3EMLKANGatedMLP(kan_model.config).to(torch.bfloat16).to(device)
        kan_model.model.layers[i].mlp = kan_mlp
        
    state_dict = torch.load(weights_path, map_location=device)
    kan_model.load_state_dict(state_dict, strict=True)
    kan_model.eval()
    
    results["EML-KAN (Uncompiled)"] = run_performance_test(
        kan_model, tokenizer, test_prompts, "EML-KAN (Uncompiled)", device
    )
    
    # --------------------------------------------------------------------------
    # Configuration 4: EML-KAN Model (Compiled)
    # --------------------------------------------------------------------------
    print("\nCompiling EML-KAN Hybrid model (torch.compile)...")
    t_comp_start = time.time()
    compiled_kan = torch.compile(kan_model)
    # Trigger compilation
    with torch.no_grad():
        _ = compiled_kan.generate(**warmup_inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    print(f"KAN compilation time: {time.time() - t_comp_start:.2f} seconds.")
    
    results["EML-KAN (Compiled)"] = run_performance_test(
        compiled_kan, tokenizer, test_prompts, "EML-KAN (Compiled)", device
    )
    
    # --------------------------------------------------------------------------
    # Output Comparison Table
    # --------------------------------------------------------------------------
    print("\n" + "="*95)
    print("                               GEMMA-3 VS EML-KAN BENCHMARK REPORT")
    print("="*95)
    print(f"{'Configuration':<30} | {'Avg TTFT (ms)':<15} | {'Avg Throughput (t/s)':<22} | {'Avg Latency (s)':<15}")
    print("-"*95)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['avg_ttft']:<15.1f} | {metrics['avg_throughput']:<22.1f} | {metrics['avg_latency']:<15.2f}")
    print("="*95)

if __name__ == "__main__":
    main()
