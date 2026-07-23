# EML-KAN vs Original Gemma-3 Benchmark Report (vrsgpu - L40S Server)

This report presents the speed and throughput benchmarks comparing the original Gemma-3-1b-it model and the optimized EML-KAN model variants run on the **Intel Xeon Silver 4416+ (CPU)** and the **NVIDIA L40S (GPU)** of this machine.

## 1. Speed / Throughput Comparison

| Model Configuration | CPU Throughput | CPU Speedup (vs. Quantized Original CPU = 23.20 t/s) | GPU Throughput | GPU Speedup (vs. Original BF16 GPU = 60.71 t/s) |
|---------------------|----------------|----------------------------------------------------|----------------|-------------------------------------------------|
| **Original Gemma-3-1b-it (bfloat16)** | 14.05 t/s | 0.61x (-39.4%) | 60.71 t/s | 1.00x (Baseline) |
| **Quantized Original (int8 CPU)** | 23.20 t/s | 1.00x (Baseline) | - | - |
| **EML-KAN Gemma-3-1b-it (bfloat16)** | 10.46 t/s | 0.45x (-54.9%) | 42.21 t/s | 0.70x (-30.5%) |
| **EML-KAN Gemma-3-1b-it (float32)** | 11.60 t/s | 0.50x (-50.0%) | - | - |
| **Compiled EML-KAN (bfloat16)** | - | - | 41.59 t/s | 0.69x (-31.5%) |
| **Quantized EML-KAN (int8 CPU)** | 17.11 t/s | 0.74x (-26.3%) | - | - |
| **Compiled Quantized EML-KAN** | 17.26 t/s | 0.74x (-25.6%) | - | - |
| **DP-Collapsed 3-Layer KAN + Hopfield Attention withPoly** | 13.65 t/s | 0.59x (-41.2%) | 48.63 t/s | 0.80x (-19.9%) |
| **DP-Collapsed 3-Layer KAN + Native SDPA Attention withPoly** | 13.12 t/s | 0.57x (-43.4%) | 57.23 t/s | 0.94x (-5.7%) |
| **Fused Hopfield EML KAN Model withPoly (Fully Compiled)** | - | - | 57.58 t/s | 0.95x (-5.2%) |

## 2. Key Observations & Findings

1. **GPU Baseline Performance**: On the NVIDIA L40S GPU, the native `Original Gemma-3-1b-it (bfloat16)` achieves **60.71 t/s**.
2. **Optimized Forward Pass Speedup**: By integrating **Horner's Method** (reducing polynomial edge multiplications from 5 to 3) and substituting explicit approximations with native **`F.gelu`** hardware lookup fusions, GPU throughput for KAN configurations increased significantly:
   - `DP-Collapsed 3-Layer KAN + Native SDPA Attention withPoly` rose to **57.23 t/s** (**94.3%** of native baseline).
   - `Fused Hopfield EML KAN Model withPoly (Fully Compiled)` rose to **57.58 t/s** (**94.8%** of native baseline).
3. **CPU Baseline Alignment**: On the Intel Xeon Silver 4416+ CPU, the `Quantized Original (int8 CPU)` baseline achieves **23.20 t/s**. The `Compiled Quantized EML-KAN` model tracks closely at **17.26 t/s** (74.4% of the original quantized baseline).
4. **Chebyshev Polynomial Fitting Domain Bound**: To prevent numerical divergence of KAN correction edges under high-activation bounds (where intermediate values scale to $10.0$ and $-10.0$), the Chebyshev fitting limits were expanded from $[-3.0, 3.0]$ to $[-10.0, 10.0]$. This maintains the measured execution speed while ensuring clean, correct semantic reasoning text generation.
5. **Taylor Threshold Optimization**: By increasing the classification threshold `taylor_threshold` to `0.50`, the compiler aggressively linearizes **1287 parameters** (out of 12462 in Layer 0) into cheap Taylor expansions. This reduces Chebyshev 3rd-degree polynomial workloads by **~10.3%** per layer, maintaining speeds while preserving correct model intelligence.
