import torch
import time
import numpy as np
from src.model import SmolVLA

def benchmark_inference(num_iters=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmolVLA()
    
    # Use the same optimizations as training
    torch.set_float32_matmul_precision('high')
    if hasattr(torch, "compile"):
        print("Compiling model for benchmark...")
        model = torch.compile(model)
        
    model.to(device)
    model.eval()
    
    # Dummy inputs
    vision = torch.randn(1, 768).to(device)
    state = torch.randn(1, 4).to(device)
    input_ids = torch.randint(0, 48000, (1, 20)).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(vision, state, input_ids)
            
    # Benchmark
    print(f"Running benchmark for {num_iters} iterations...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(vision, state, input_ids)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iters
    hz = 1.0 / avg_time
    
    print(f"Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"Inference Speed: {hz:.2f} Hz")
    
    return avg_time, hz

if __name__ == "__main__":
    benchmark_inference()
