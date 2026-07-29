# EML-KAN vs Original Gemma-3 Benchmark Report (vrsgpu - L40S Server)

This report presents the speed and throughput benchmarks comparing the original Gemma-3-1b-it model and the optimized EML-KAN model variants run on the **Intel Xeon Silver 4416+ (CPU)** and the **NVIDIA L40S (GPU)** of this machine.

## 1. Speed / Throughput Comparison

| Model Configuration | CPU Throughput | CPU Speedup (vs. Quantized Original CPU = 23.71 t/s) | GPU Throughput | GPU Speedup (vs. Original BF16 GPU = 60.17 t/s) |
|---------------------|----------------|----------------------------------------------------|----------------|-------------------------------------------------|
| **Original Gemma-3-1b-it (bfloat16)** | 13.82 t/s | 0.58x (-41.7%) | 60.17 t/s | 1.00x (Baseline) |
| **Quantized Original (int8 CPU)** | 23.71 t/s | 1.00x (Baseline) | - | - |
| **EML-KAN Gemma-3-1b-it (bfloat16)** | 11.52 t/s | 0.49x (-51.4%) | 40.68 t/s | 0.68x (-32.4%) |
| **EML-KAN Gemma-3-1b-it (float32)** | 11.16 t/s | 0.47x (-52.9%) | - | - |
| **Compiled EML-KAN (bfloat16)** | - | - | 40.67 t/s | 0.68x (-32.4%) |
| **Quantized EML-KAN (int8 CPU)** | 16.73 t/s | 0.71x (-29.4%) | - | - |
| **Compiled Quantized EML-KAN** | 16.52 t/s | 0.70x (-30.3%) | - | - |
| **Fused Hopfield EML KAN Model withPoly (Fully Compiled)** | - | - | 58.59 t/s | 0.97x (-2.6%) |

## 2. Key Observations & Findings

1. **GPU Baseline Performance**: On the NVIDIA L40S GPU, the native `Original Gemma-3-1b-it (bfloat16)` achieves **60.17 t/s**.
2. **Soft-Bounded EML KAN Architecture**: By replacing the hard `torch.clamp(..., -10.0, 10.0)` EMLCorrection arguments with a smooth soft-clipping function:
   $$\text{arg}_x = 3.0 \times \tanh\left(\frac{ax+b}{3.0}\right)$$
   we mathematically guarantee that inputs to the exponential functions never exceed the stable range $[-3.0, 3.0]$. This avoids training instabilities, removes sharp "elbows" (saturation limits), and allows highly accurate Chebyshev polynomial approximations.
3. **Tight Domain Compilation**: Under the tanh-bounded design, the Taylor-Polynomial Hybrid Compiler compiles the model with a tight `domain_bound=3.0`.
   - Taylor (linearized) components jump to **8617 parameters** (~71% of EML components) per layer.
   - The compiled model achieves **58.59 t/s** (**97.4%** of native GPU speed) and retains **100% correct, sensible math reasoning, coding, and logical outputs**.
4. **Lossless EML Grammar Folding**: By exploiting the additive EML grammar:
   $$\text{gate\_out} = \text{gate\_linear} + p_0 + p_1 \cdot \text{gate\_linear} + p_2 \cdot \text{gate\_linear}^2 + p_3 \cdot \text{gate\_linear}^3$$
   we algebraically fold the linear identity term $1.0$ directly into the polynomial's linear coefficient ($p'_1 = p_1 + 1.0$) during compilation. This completely eliminates one tensor addition operation (`gate_linear + eml_corr`) in the forward pass of every layer, achieving mathematical lossless compute reduction.
5. **Multiplicative Cross-Term Decoupling**: By proving that the EML cross-term coefficient $c_{1,1}$ in $c_{1,1} \cdot u \cdot v$ is zero, the compiler decouples the bivariate fitting into independent, parallel 1D additive paths ($P_A(u) + P_B(v)$). This splits the Chebyshev minimax polynomial fit step into parallel univariate processes, ensuring lossless mathematical decoupling of the exponential and logarithmic branches.


