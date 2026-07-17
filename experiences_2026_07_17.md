# EML-KAN Optimization Experiences, Math, and Learnings (July 17, 2026)

This document details the engineering journey, failures, mathematical proofs, implementation strategies, and benchmarked speedups for optimizing the EML-KAN (Exponential-Minus-Log Kolmogorov-Arnold Network) layer on Intel CPU architecture.

---

## 1. Files Touched & Created
* [eml_unified_optimizer.py](file:///C:/Users/karthikkrazy/Documents/antigravity/splendid-bohr/eml_unified_optimizer.py): Vectorized EML-KAN DAG compiler (PyTorch CPU edition).
* [eml_symbolic_regression.py](file:///C:/Users/karthikkrazy/Documents/antigravity/splendid-bohr/eml_symbolic_regression.py): Distributive pre-summed polynomial regression compiler.
* [eml_taylor_sharing_compiler.py](file:///C:/Users/karthikkrazy/Documents/antigravity/splendid-bohr/eml_taylor_sharing_compiler.py): Compiler implementing Taylor linearization and scale sharing.
* [laptop_EML_KAN_vs_ORIGINAL.md](file:///C:/Users/karthikkrazy/Documents/antigravity/splendid-bohr/laptop_EML_KAN_vs_ORIGINAL.md): Central benchmark report.
* [full_model_polynomial_benchmark.py](file:///C:/Users/karthikkrazy/Documents/antigravity/splendid-bohr/full_model_polynomial_benchmark.py): Standardized whole-model benchmark script for polynomial KAN configuration.
* [full_model_taylor_sharing_benchmark.py](file:///C:/Users/karthikkrazy/Documents/antigravity/splendid-bohr/full_model_taylor_sharing_benchmark.py): Standardized whole-model benchmark script for Taylor-sharing KAN configuration.

---

## 2. Mathematical Formulations & Optimization Strategies

### A. The Core EML-KAN Equation
The activation function $f_i(x)$ for neuron $i$ across $K = 4$ mixture components is defined as:
$$f_i(x) = \sum_{k=1}^{K} w_{i,k} \cdot \left(\text{exp}(a_{i,k}x + b_{i,k}) - \text{log}(\text{softplus}(c_{i,k}x + d_{i,k}) + \epsilon)\right)$$

### B. Distributive Compiled KAN Polynomial
#### **The Math Behind It:**
Instead of evaluating the expensive transcendental equations (exp, log, softplus) for each component $k$, we approximate the $k$-th component using a 3rd-degree polynomial:
$$P_{i,k}(x) = p_{0, i, k} + p_{1, i, k}x + p_{2, i, k}x^2 + p_{3, i, k}x^3$$
Substituting this into the mixture summation:
$$f_i(x) \approx \sum_{k=1}^{K} \left( p_{0, i, k} + p_{1, i, k}x + p_{2, i, k}x^2 + p_{3, i, k}x^3 \right)$$

By the **distributive property of addition**, we factor out $x, x^2, x^3$:
$$f_i(x) \approx \left(\sum_{k=1}^{K} p_{0, i, k}\right) + \left(\sum_{k=1}^{K} p_{1, i, k}\right)x + \left(\sum_{k=1}^{K} p_{2, i, k}\right)x^2 + \left(\sum_{k=1}^{K} p_{3, i, k}\right)x^3$$

We define pre-summed, 1D coefficients for each neuron $i$:
$$C_{m, i} = \sum_{k=1}^{K} p_{m, i, k} \quad (\text{for } m \in \{0, 1, 2, 3\})$$

#### **The Runtime Advantage:**
* **Dimensional Collapse**: The component dimension $K$ is collapsed pre-execution. We compile a simple 1D vector of coefficients per power.
* **No transcendentals**: Zero `exp`, `log`, or `softplus` evaluations.
* **FP32 Performance**: `2.69 t/s` (+58.1% speedup).
* **Quantized Compiled Performance**: **`7.25 t/s`** (+3.66x speedup over Gemma-3 original baseline).

---

### C. Taylor Linearization & Shared Scale Fusion
#### **The Math Behind It:**
1. **Taylor Linearization**: If the active bounds satisfy $|a \cdot x + b| < \text{threshold}$ and $|c \cdot x + d| < \text{threshold}$ (safe limit $= 0.08$), the KAN term is highly linear:
   $$\text{exp}(u) \approx 1 + u \quad \text{and} \quad \text{log}(\text{softplus}(v)) \approx -0.3665 + 0.7213 \cdot v$$
   The entire component collapses to:
   $$f_{i,k}(x) \approx w_{i,k} \cdot \left[ (1.3665 + b - 0.7213 \cdot d) + (a - 0.7213 \cdot c) \cdot x \right]$$
2. **Shared Scale Fusion**: If $|a - c| < \text{threshold}$ (safe limit $= 0.03$), we set $c = a$ to share the scale product:
   $$u = a \cdot x$$
   This eliminates one floating-point multiplication per component.

#### **Performance:**
* **FP32 Performance**: `2.70 t/s` (+58.8% speedup).
* **Quantized Compiled Performance**: **`5.25 t/s`** (+2.65x speedup over baseline).

---

## 3. Engineering Failures & Key Learnings

### Failure 1: The `np.where` Eager Execution Trap
* **Symptom**: Initial constant-folding attempts using NumPy's `np.where` actually ran slower than the un-folded vectorized execution.
* **Cause**: NumPy evaluates both arguments in `np.where(cond, x, y)` eagerly. The expensive `np.exp` or `np.log` was still computed for all elements, and the conditional masks added addressing overhead.
* **Learning**: Vectorized SIMD code is faster when running contiguous math; adding boolean branch masking at runtime hurts instruction cache efficiency.

### Failure 2: Dynamic Quantization Bypassing Tensors
* **Symptom**: Early PyTorch Compiled DAGs did not show the expected quantized speedups.
* **Cause**: PyTorch's `quantize_dynamic(model, {nn.Linear})` checks the module tree for instances of `nn.Linear`. Because the compiler represented weight projections as dictionary lookups evaluated via `torch.matmul`, they were bypassed and ran in full FP32.
* **Learning**: Wrap variables in standard `nn.Linear` modules inside custom classes to allow the quantization engine to automatically inspect, target, and quantize the parameters.

### Failure 3: `cl.exe` Dependency in Standalone `torch.compile`
* **Symptom**: Compiling the custom `eval_dag` function directly threw `RuntimeError: Compiler: cl is not found`.
* **Cause**: Windows requires the MSVC C++ compiler to generate C++ CPU kernel loops for custom indexing patterns (`index_add_`).
* **Learning**: Avoid compiling custom loose functions directly on Windows. Wrapping them in a subclass of `nn.Module` and compiling the model container allows Inductor to fallback compile cleanly.

---

## 4. Benchmark Leaderboard Summary (CPU)

| Model Configuration | Precision / Type | Throughput (Tokens/sec) | Speed vs. Original Baseline |
|---------------------|------------------|-------------------------|-----------------------------|
| **Original Gemma-3-1b-it** | bfloat16 | 1.98 t/s | 1.00x |
| **EML-KAN Gemma-3-1b-it** | bfloat16 | 1.41 t/s | 0.71x |
| **EML-KAN Gemma-3-1b-it** | float32 | 1.70 t/s | 0.86x |
| **Quantized Compiled Taylor-Sharing KAN** | int8 dynamic + FP32 | **5.25 t/s** | **2.65x** |
| **Quantized Compiled Polynomial EML-KAN** | int8 dynamic + FP32 | **7.25 t/s** | **3.66x** |
