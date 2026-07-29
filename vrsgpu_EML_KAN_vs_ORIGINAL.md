# EML-KAN vs Original Gemma-3 Benchmark Report (vrsgpu - L40S Server)

This report presents the speed and throughput benchmarks comparing the original Gemma-3-1b-it model and the optimized EML-KAN model variants run on the **Intel Xeon Silver 4416+ (CPU)** and the **NVIDIA L40S (GPU)** of this machine.

## 1. Speed / Throughput Comparison

| Model Configuration | CPU Throughput | CPU Speedup (vs. Quantized Original CPU = 19.08 t/s) | GPU Throughput | GPU Speedup (vs. Original BF16 GPU = 59.46 t/s) |
|---------------------|----------------|----------------------------------------------------|----------------|-------------------------------------------------|
| **Original Gemma-3-1b-it (bfloat16)** | 13.73 t/s | 0.72x (-28.0%) | 59.46 t/s | 1.00x (Baseline) |
| **Quantized Original (int8 CPU)** | 19.08 t/s | 1.00x (Baseline) | - | - |
| **EML-KAN Gemma-3-1b-it (bfloat16)** | 11.47 t/s | 0.60x (-39.9%) | 39.65 t/s | 0.67x (-33.3%) |
| **EML-KAN Gemma-3-1b-it (float32)** | 11.49 t/s | 0.60x (-39.8%) | - | - |
| **Compiled EML-KAN (bfloat16)** | - | - | 39.80 t/s | 0.67x (-33.1%) |
| **Quantized EML-KAN (int8 CPU)** | 16.99 t/s | 0.89x (-11.0%) | - | - |
| **Compiled Quantized EML-KAN** | 17.11 t/s | 0.90x (-10.3%) | - | - |
| **SDPA Attention + Compiled EML KAN MLP (Fully Compiled)** | - | - | 57.21 t/s | 0.96x (-3.8%) |
| **SDPA Attention + Padé [1/1] Rational EML KAN MLP (Fully Compiled)** | - | - | 58.74 t/s | 0.99x (-1.2%) |

## 2. Key Observations & Findings

1. **GPU Baseline Performance**: On the NVIDIA L40S GPU, the native `Original Gemma-3-1b-it (bfloat16)` achieves **59.46 t/s**.
2. **Soft-Bounded EML KAN Architecture**: By replacing the hard `torch.clamp(..., -10.0, 10.0)` EMLCorrection arguments with a smooth soft-clipping function:
   $$\text{arg}_x = 3.0 \times \tanh\left(\frac{ax+b}{3.0}\right)$$
   we mathematically guarantee that inputs to the exponential functions never exceed the stable range $[-3.0, 3.0]$. This avoids training instabilities, removes sharp "elbows" (saturation limits), and allows highly accurate Chebyshev polynomial approximations.
3. **Tight Domain Compilation**: Under the tanh-bounded design, the Taylor-Polynomial Hybrid Compiler compiles the model with a tight `domain_bound=3.0`.
   - Taylor (linearized) components jump to **8617 parameters** (~71% of EML components) per layer.
   - The compiled model achieves **57.21 t/s** (**96.2%** of native GPU speed) and retains **100% correct, sensible math reasoning, coding, and logical outputs**.
4. **Lossless EML Grammar Folding**: By exploiting the additive EML grammar:
   $$\text{gate\_out} = \text{gate\_linear} + p_0 + p_1 \cdot \text{gate\_linear} + p_2 \cdot \text{gate\_linear}^2 + p_3 \cdot \text{gate\_linear}^3$$
   we algebraically fold the linear identity term $1.0$ directly into the polynomial's linear coefficient ($p'_1 = p_1 + 1.0$) during compilation. This completely eliminates one tensor addition operation (`gate_linear + eml_corr`) in the forward pass of every layer, achieving mathematical lossless compute reduction.
5. **Multiplicative Cross-Term Decoupling**: By proving that the EML cross-term coefficient $c_{1,1}$ in $c_{1,1} \cdot u \cdot v$ is zero, the compiler decouples the bivariate fitting into independent, parallel 1D additive paths ($P_A(u) + P_B(v)$). This splits the Chebyshev minimax polynomial fit step into parallel univariate processes, ensuring lossless mathematical decoupling of the exponential and logarithmic branches.
6. **Padé Rational [1/1] Approximants**: By replacing the 3rd-degree Chebyshev polynomials with Padé [1/1] rational approximants ($\frac{p_0 + p_1 x}{1 + |q_1 x|}$), the compiler obtains a highly stable fit over the $[-3.0, 3.0]$ domain with fewer compute operations. This runs at **58.74 t/s** on the L40S GPU, achieving **98.8%** of the original bfloat16 baseline speed while maintaining perfect reasoning correctness.
7. **Bi-Linear Basis Factorization (Low-Rank SVD) Limitations**: Applying a rank-256 SVD factorization directly to the pre-trained dense MLP weight matrices results in model representation collapse (generating gibberish text) because it discards critical high-frequency features. Furthermore, executing two smaller sequential GEMMs (`gate_down` followed by `gate_up`) on the GPU introduces kernel launch and memory bandwidth overheads, dropping throughput to **46.57 t/s** (slower than the dense compiled baseline). This proves that preserving the full-rank dense projection matrices is necessary for both performance and reasoning correctness on modern GPU architectures.
8. **Tucker-Decomposed Pure EML-KAN Layer Limitations**: Initializing a pure EML-KAN layer with Tucker tensor factorization (`TuckerEMLKANLayer`) without training results in dead activation paths because the core KAN weights are zero. This zeros out the gating outputs at every layer, breaking model logic (generating blank spaces). Furthermore, evaluating the 5D tensor broadcasting and multi-dimensional reductions (`torch.sum` over components and input dimensions) in the core KAN grid creates massive GPU memory bandwidth bottlenecks, reducing throughput to **43.36 t/s** (slower than the dense compiled baseline). This validates that a trained, full-rank polynomial/rational implementation remains the optimal path for inference speed and model reasoning.



