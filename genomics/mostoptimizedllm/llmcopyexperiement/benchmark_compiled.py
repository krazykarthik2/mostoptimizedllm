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
    
    # Warmup and Compile trigger
    print(f"  Warmup and compiling model (first generation run on {device})...")
    t0 = time.time()
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
            torch.cuda.synchronize()
    compilation_time = time.time() - t0
    print(f"  Compilation & Warmup took {compilation_time:.2f} seconds.")
            
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

def run_compiled_benchmarks(model_id, weights_path, device="cuda"):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    prompt = "Write a short summary about the importance of scientific research."
    print(f"Benchmark Prompt: '{prompt}'")
    
    results = {}
    
    # 1. Benchmark torch.compile of Fitted EML-KAN Gemma-3 on GPU
    print("\n--- BENCHMARKING COMPILED EML-KAN GEMMA-3 (GPU) ---")
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
        
        # Compile model
        print("Compiling EML-KAN model with torch.compile(mode='reduce-overhead') on GPU...")
        compiled_model_kan = torch.compile(model_kan, mode="reduce-overhead")
        
        tps_kan, lat_kan, num_toks = measure_tps(compiled_model_kan, tokenizer, prompt, device)
        print(f"Compiled EML-KAN Model (GPU): {tps_kan:.2f} tokens/sec | Avg Latency: {lat_kan:.2f} ms for {num_toks} tokens")
        results["Compiled EML-KAN (GPU)"] = (tps_kan, lat_kan)
        
        # Free GPU memory
        del model_kan
        del compiled_model_kan
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to benchmark compiled EML-KAN model on GPU: {e}")
        
    # 2. Benchmark torch.compile of Fitted EML-KAN Gemma-3 on CPU
    print("\n--- BENCHMARKING COMPILED EML-KAN GEMMA-3 (CPU) ---")
    try:
        # Re-load for CPU benchmarking in float32
        model_cpu = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.float32
        )
        for i in range(model_cpu.config.num_hidden_layers):
            model_cpu.model.layers[i].mlp = Gemma3EMLKANMLP(model_cpu.config).to(torch.float32)
            
        state_dict_cpu = torch.load(weights_path, map_location="cpu")
        # Convert state dict to float32
        for k in state_dict_cpu.keys():
            state_dict_cpu[k] = state_dict_cpu[k].float()
        model_cpu.load_state_dict(state_dict_cpu)
        model_cpu.eval()
        
        # Compile model on CPU
        # mode="reduce-overhead" is supported on CPU since PyTorch 2.1
        print("Compiling EML-KAN model with torch.compile(mode='reduce-overhead') on CPU...")
        compiled_model_cpu = torch.compile(model_cpu, mode="reduce-overhead")
        
        # Run CPU benchmark with fewer tokens to run quickly
        tps_cpu, lat_cpu, num_toks = measure_tps(compiled_model_cpu, tokenizer, prompt, "cpu", max_new_tokens=20, num_runs=2)
        print(f"Compiled EML-KAN Model (CPU): {tps_cpu:.2f} tokens/sec | Avg Latency: {lat_cpu:.2f} ms for {num_toks} tokens")
        results["Compiled EML-KAN (CPU)"] = (tps_cpu, lat_cpu)
    except Exception as e:
        print(f"Failed to benchmark compiled EML-KAN model on CPU: {e}")
        
    # Print final results summary table
    print("\n" + "="*60)
    print("                COMPILED TPS PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model Configuration':<30} | {'Throughput (TPS)':<18} | {'Avg Latency (ms)':<18}")
    print("-"*72)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics[0]:>14.2f} tps | {metrics[1]:>14.2f} ms")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark compiled model generation speed")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="Original Gemma model ID")
    parser.add_argument("--weights_path", type=str, default="gemma3_eml_kan/model_state_tuned.pt", help="Path to EML-KAN model weights")
    args = parser.parse_args()
    
    run_compiled_benchmarks(args.model_id, args.weights_path)
