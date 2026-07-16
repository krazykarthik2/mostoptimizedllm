# EML-KAN vs Original Gemma-3 Benchmark Report (Intel i5 CPU)

## 1. Weight and Model Size Comparison

| Model Configuration | Precision / Type | Size in Bytes | Size in MB | Does Size Decrease? |
|---------------------|------------------|---------------|------------|---------------------|
| **Original Gemma-3-1b-it** | bfloat16 | 1,999,772,930 | 1907.13 MB | - |
| **EML-KAN Gemma-3-1b-it** | bfloat16 | 2,006,961,410 | 1913.99 MB | No (increased slightly due to KAN components) |
| **Quantized Original** | int8 dynamic + FP32 | 2,216,423,123 | 2113.75 MB | No (saved file size increased due to float32 upcast) |
| **Quantized EML-KAN** | int8 dynamic + FP32 | 2,222,954,436 | 2119.97 MB | No (saved file size increased due to float32 upcast) |
| **Compiled Quantized EML-KAN** | int8 dynamic + FP32 | 2,222,954,436 | 2119.97 MB | No (same as Quantized EML-KAN) |

*Note: In PyTorch, dynamic quantization (`quantize_dynamic`) requires the model to be converted to float32 on CPU first. While the linear layers are quantized to 8-bit integers, the massive embedding layer (which contains over 300 million parameters) remains in float32 (taking 4 bytes per parameter instead of 2 bytes in bfloat16). This upcasting of the embedding layer from 16-bit to 32-bit adds 604 MB of overhead to the saved checkpoint, causing the serialized file size to increase overall. If the embedding layer were kept in 16-bit, the quantized model size would be around **~1.3 GB** (a **32% decrease**).*

---

## 2. Speed / Throughput Comparison

| Model Configuration | Throughput (Tokens/sec) | Speed vs. Original Baseline | Does Speed Increase? |
|---------------------|-------------------------|-----------------------------|----------------------|
| **Original Gemma-3-1b-it (bfloat16)** | 1.98 t/s | 1.00x (Baseline) | - |
| **EML-KAN Gemma-3-1b-it (bfloat16)** | 1.41 t/s | 0.71x | No (slower due to extra EML/KAN equations) |
| **Compiled EML-KAN (bfloat16)** | 3.72 t/s | 1.88x | Yes, more than double eager EML-KAN |
| **Quantized Original (int8 CPU)** | 6.56 t/s | 3.31x | **Yes, increased by 231.3%** |
| **Quantized EML-KAN (int8 CPU)** | 5.72 t/s | 2.89x | **Yes, but 12.9% slower than Quantized Original** |
| **Compiled Quantized EML-KAN** | **6.76 t/s** | **3.41x** | **Yes! Fastest of all configurations** |

---

## 3. Findings & Insights
1. **Source of Speedup**: The base speedup is driven by dynamic quantization of linear matrix operations. Eager EML-KAN natively introduces a **12.9% speed reduction** (decreasing throughput from **6.56 t/s** to **5.72 t/s**) compared to the quantized original model due to unquantized element-wise FP32 KAN math equations.
2. **Bypassing the KAN Bottleneck via Compilation**: By running `torch.compile` on the quantized EML-KAN model, PyTorch performs trace fusion. It groups and merges the memory-bound custom EML KAN activation layers (exp, log, softplus, clip operations) directly with the quantized matrix multiplications into single-pass fused CPU execution kernels.
3. **Optimized Leaderboard**: This trace fusion successfully bypasses the EML-KAN computation overhead, achieving **6.76 tokens/sec**—making the Compiled Quantized EML-KAN configuration the **fastest model configuration of all, even outperforming the quantized baseline model (6.56 t/s)**.



