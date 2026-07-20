# EML-KAN vs Original Gemma-3 Benchmark Report (vrsgpu - L40S Server)

This report presents the speed and throughput benchmarks comparing the original Gemma-3-1b-it model and the optimized EML-KAN model variants run on the **Intel Xeon Silver 4416+ (CPU)** and the **NVIDIA L40S (GPU)** of this machine.

## 1. Speed / Throughput Comparison

| Model Configuration | CPU Throughput | CPU Speedup (vs. Quantized Original CPU = 22.86 t/s) | GPU Throughput | GPU Speedup (vs. Original BF16 GPU = 30.43 t/s) |
|---------------------|----------------|----------------------------------------------------|----------------|-------------------------------------------------|
| **Original Gemma-3-1b-it (bfloat16)** | 16.81 t/s | 0.74x (-26.5%) | 30.43 t/s | 1.00x (Baseline) |
| **Quantized Original (int8 CPU)** | 22.86 t/s | 1.00x (Baseline) | - | - |
| **EML-KAN Gemma-3-1b-it (bfloat16)** | 13.20 t/s | 0.58x (-42.3%) | 40.77 t/s | 1.34x (+34.0%) |
| **EML-KAN Gemma-3-1b-it (float32)** | 11.87 t/s | 0.52x (-48.1%) | - | - |
| **Compiled EML-KAN (bfloat16)** | - | - | 39.42 t/s | 1.30x (+29.5%) |
| **Quantized EML-KAN (int8 CPU)** | 16.61 t/s | 0.73x (-27.3%) | - | - |
| **Compiled Quantized EML-KAN** | 16.77 t/s | 0.73x (-26.6%) | - | - |
| **Collapsed 2-Layer KAN + Hopfield Attention withPoly** | 22.02 t/s | 0.96x (-3.7%) | 23.67 t/s | 0.78x (-22.2%) |
| **DP-Collapsed 3-Layer KAN + Hopfield Attention withPoly** | 21.96 t/s | 0.96x (-3.9%) | 44.96 t/s | 1.48x (+47.8%) |
| **DP-Collapsed 3-Layer KAN + Native SDPA Attention withPoly** | 24.28 t/s | 1.06x (+6.2%) | 53.63 t/s | 1.76x (+76.2%) |
| **Fused Hopfield EML KAN Model withPoly (Fully Compiled)** | 20.18 t/s | 0.88x (-11.7%) | 44.48 t/s | 1.46x (+46.2%) |
| **Query-Cancelled Hopfield EML KAN Model** | 17.48 t/s | 0.76x (-23.5%) | 49.08 t/s | 1.61x (+61.3%) |
| **Fused GELU GLU + Hopfield Attention Model withPoly** | 21.89 t/s | 0.96x (-4.2%) | 44.91 t/s | 1.48x (+47.6%) |

## 2. Key Observations & Findings

1. **GPU Baseline Performance**: On the NVIDIA L40S GPU, the native `Original Gemma-3-1b-it (bfloat16)` serves as the fastest original baseline, achieving **30.43 t/s**.
2. **GPU Performance and KAN Fusions**: The `DP-Collapsed 3-Layer KAN + Native SDPA Attention withPoly` configuration reaches **53.63 t/s** on GPU, representing a **1.76x speedup** (+76.2% improvement) over the GPU baseline.
3. **CPU Baseline Alignment**: On the Intel Xeon Silver 4416+ CPU, the `Quantized Original (int8 CPU)` baseline achieves **22.86 t/s**. The `DP-Collapsed 3-Layer KAN + Native SDPA Attention withPoly` model achieves **24.28 t/s** (+6.2% speedup), which surpasses the quantized original baseline on CPU.
4. **Quantization Comparison**: INT8 quantized execution on GPU is slower than native BF16 due to dynamic casting and scaling overheads during autoregressive sequence generation.
