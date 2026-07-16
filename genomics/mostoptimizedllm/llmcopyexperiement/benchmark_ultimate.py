import os
import time
import torch
from model import Gemma3EMLKANGatedMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("="*80)
    print("                 EML-KAN TRANSFORMER BENCHMARK FRAMEWORK")
    print("="*80)
    
    # 1. Benchmark Cold Boot Time
    t_boot_start = time.time()
    
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state_regularized.pt"
    device = "cuda:0"
    
    print("Importing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading base model {model_id} into GPU memory...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    
    print("Executing EML-KAN MLP architectural swap...")
    for i in range(model.config.num_hidden_layers):
        kan_mlp = Gemma3EMLKANGatedMLP(model.config).to(torch.bfloat16).to(device)
        model.model.layers[i].mlp = kan_mlp
        
    print(f"Loading calibrated weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Run a warm-up generation to compile paths/CUDA kernels
    print("Running warm-up generation...")
    warmup_inputs = tokenizer("Warm up", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
        
    t_boot_end = time.time()
    cold_boot_time = t_boot_end - t_boot_start
    print(f"COLD BOOT TIME: {cold_boot_time:.2f} seconds.")
    print("="*80)
    
    # 2. Run Benchmarks on OOD Prompts
    test_prompts = [
        "If a train travels 60 miles per hour, how far will it travel in 2.5 hours? Explain your reasoning step-by-step.",
        "A father has 4 daughters. Each daughter has a brother. How many children does the father have in total? Explain your reasoning.",
        "Write a python function to check if a given integer is a prime number.",
        "What is the height of Mount Everest?"
    ]
    
    latencies = []
    ttfts = []
    throughputs = []
    total_tokens_gen = 0
    
    print("\nRunning Inference Performance & Intelligence Tests...")
    
    for idx, p in enumerate(test_prompts):
        messages = [{"role": "user", "content": p}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        
        # Measure TTFT (Time to First Token)
        t_start = time.time()
        with torch.no_grad():
            # Generate first token
            outputs_first = model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id
            )
        t_first = time.time()
        ttft = (t_first - t_start) * 1000.0  # convert to ms
        ttfts.append(ttft)
        
        # Measure full generation
        t_gen_start = time.time()
        with torch.no_grad():
            outputs_full = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        t_gen_end = time.time()
        
        latency = t_gen_end - t_gen_start
        latencies.append(latency)
        
        # Calculate tokens generated
        total_tokens = outputs_full[0].shape[0]
        tokens_generated = total_tokens - input_len
        total_tokens_gen += tokens_generated
        
        throughput = tokens_generated / latency if latency > 0 else 0
        throughputs.append(throughput)
        
        response = tokenizer.decode(outputs_full[0][input_len:], skip_special_tokens=True).strip()
        
        print(f"\n[{idx+1}] Prompt: {p}")
        print("-"*80)
        print(response)
        print("-"*80)
        print(f"Metrics: TTFT = {ttft:.1f}ms | Latency = {latency:.2f}s | Throughput = {throughput:.1f} tokens/s | Generated = {tokens_generated} tokens")
        print("="*80)
        
    # 3. Print Benchmark Summary Table
    avg_ttft = sum(ttfts) / len(ttfts)
    avg_throughput = sum(throughputs) / len(throughputs)
    avg_latency = sum(latencies) / len(latencies)
    
    print("\n" + "="*80)
    print("                     BENCHMARK METRICS SUMMARY")
    print("="*80)
    print(f"Cold Boot Time:         {cold_boot_time:.2f} seconds")
    print(f"Average TTFT:           {avg_ttft:.1f} ms")
    print(f"Average Generation:     {avg_latency:.2f} seconds")
    print(f"Average Throughput:     {avg_throughput:.1f} tokens/second")
    print(f"Total Tokens Generated: {total_tokens_gen} tokens")
    print("="*80)

if __name__ == "__main__":
    main()
