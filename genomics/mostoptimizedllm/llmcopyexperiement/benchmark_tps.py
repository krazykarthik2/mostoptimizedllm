import os
import gc
import time
import torch
import torch.nn as nn
import argparse
from model import Gemma3EMLKANMLP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def measure_tps(model, tokenizer, prompt, device, max_new_tokens=50, num_runs=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
            torch.cuda.synchronize()
            
    # Benchmark runs
    total_time = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for run in range(num_runs):
            t0 = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            
            generated_tokens = outputs.shape[1] - prompt_len
            total_time += dt
            total_tokens += generated_tokens
            
    avg_tps = total_tokens / total_time
    avg_latency = (total_time / num_runs) * 1000.0 # in ms
    return avg_tps, avg_latency, total_tokens // num_runs

def run_benchmarks(model_id, weights_path, device="cuda"):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    prompt = "Write a short summary about the importance of scientific research."
    print(f"Benchmark Prompt: '{prompt}'")
    
    results = {}
    
    # 1. Benchmark Original Baseline Gemma-3-1b-it on GPU
    print("\n--- BENCHMARKING ORIGINAL GEMMA-3-1B-IT (GPU) ---")
    try:
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.bfloat16
        ).to(device)
        model_orig.eval()
        
        tps_orig, lat_orig, num_toks = measure_tps(model_orig, tokenizer, prompt, device)
        print(f"Original Model: {tps_orig:.2f} tokens/sec | Avg Latency: {lat_orig:.2f} ms for {num_toks} tokens")
        results["Original (GPU)"] = (tps_orig, lat_orig)
        
        # Cleanup original model to save VRAM
        del model_orig
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to benchmark original model: {e}")
        
    # 2. Benchmark Fitted EML-KAN Gemma-3 on GPU
    print("\n--- BENCHMARKING FITTED EML-KAN GEMMA-3 (GPU) ---")
    try:
        model_kan = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.bfloat16
        ).to(device)
        for i in range(model_kan.config.num_hidden_layers):
            model_kan.model.layers[i].mlp = Gemma3EMLKANMLP(model_kan.config).to(torch.bfloat16).to(device)
            
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model_kan.load_state_dict(state_dict)
        model_kan.eval()
        
        tps_kan, lat_kan, num_toks = measure_tps(model_kan, tokenizer, prompt, device)
        print(f"EML-KAN Model: {tps_kan:.2f} tokens/sec | Avg Latency: {lat_kan:.2f} ms for {num_toks} tokens")
        results["EML-KAN (GPU)"] = (tps_kan, lat_kan)
    except Exception as e:
        print(f"Failed to benchmark EML-KAN model: {e}")
        
    # 3. Benchmark Pruned + Quantized EML-KAN model on CPU
    print("\n--- BENCHMARKING COMPRESSED EML-KAN (CPU QUANTIZED) ---")
    try:
        # Move model to CPU and convert to float32
        model_cpu = model_kan.cpu().float()
        
        # Apply 50% magnitude pruning to linear weights
        with torch.no_grad():
            for name, param in model_cpu.named_parameters():
                if "linear.weight" in name or "weight_eml" in name:
                    threshold = torch.quantile(torch.abs(param), 0.5)
                    mask = torch.abs(param) >= threshold
                    param.mul_(mask.float())
                    
        # Apply 8-bit dynamic quantization
        print("Quantizing model dynamic linear ops to int8...")
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        tps_quant, lat_quant, num_toks = measure_tps(quantized_model, tokenizer, prompt, "cpu", max_new_tokens=20, num_runs=2)
        print(f"Quantized CPU Model: {tps_quant:.2f} tokens/sec | Avg Latency: {lat_quant:.2f} ms for {num_toks} tokens")
        results["Quantized EML-KAN (CPU)"] = (tps_quant, lat_quant)
    except Exception as e:
        print(f"Failed to benchmark Quantized model: {e}")
        
    # Print final results summary table
    print("\n" + "="*60)
    print("                     TPS PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Model Configuration':<30} | {'Throughput (TPS)':<18} | {'Avg Latency (ms)':<18}")
    print("-"*72)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics[0]:>14.2f} tps | {metrics[1]:>14.2f} ms")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model generation speed (Tokens Per Second)")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="Original Gemma model ID")
    parser.add_argument("--weights_path", type=str, default="gemma3_eml_kan/model_state_tuned.pt", help="Path to EML-KAN model weights")
    args = parser.parse_args()
    
    run_benchmarks(args.model_id, args.weights_path)
