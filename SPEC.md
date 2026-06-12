# SmolVLA Technical Specifications

## 1. Model Overview
*   **Architecture:** SmolVLA (inspired by π0.5).
*   **Backbone:** SmolLM2-360M (24 layers active out of 32).
*   **Total Parameters:** 291,776,064 (~292M).
*   **Trainable Parameters:** 47,942,784 (~48M).
*   **Precision:** BF16 (Training & Inference).
*   **Optimization:** `torch.compile` (Inductor), TensorFloat32 (TF32).

## 2. Input/Output Dimensions
*   **Vision Input:** 768-dim SigLIP embedding -> Projected to **8 vision tokens**.
*   **State Input:** 4-dim EEF state [x, y, z, g] -> Projected to **2 state tokens**.
*   **Language Input:** Variable length tokens (SmolLM Tokenizer).
*   **Token Sequence:** `[VISION (8)] [STATE (2)] [LANGUAGE (N)]`.
*   **Output:** 64-dim vector -> Reshaped to **16 waypoints × 4 dims** [Δx, Δy, Δz, Δg].

## 3. Scaling & Optimizations
*   **Optimizer:** Muon (MomentUm Orthogonalized by Newton-Schulz).
*   **Scale Approximation:** Fast Inverse Square Root (`0x5f3759df`) trick for optimizer scaling.
*   **Checkpointing:** Architecture-aware versioning (automatic cleanup on mismatch).
*   **Compilation:** Full graph compilation for kernel fusion and optimized memory layout.

## 4. Performance Metrics (Single Sample Inference)
Measured on **NVIDIA L4 (24GB VRAM)**:

| Metric | Value |
| :--- | :--- |
| **Inference Time** | **9.47 ms** |
| **Control Frequency** | **105.58 Hz** |
| **Peak Throughput** | **1798 samples/sec** (Batch size 128) |
| **VRAM Required** | **~1.2 GB** (Model) / **~4-8 GB** (Training) |

## 5. GPU Compatibility Estimates
*Estimated inference speeds based on L4 baseline:*

| GPU Model | Estimated Latency | Estimated Hz |
| :--- | :--- | :--- |
| **NVIDIA L4** | **9.47 ms** | **105 Hz** |
| **NVIDIA A100** | ~4-6 ms | ~150-200 Hz |
| **NVIDIA H100** | ~2-3 ms | ~300-400 Hz |
| **NVIDIA RTX 4090**| ~5-7 ms | ~140-180 Hz |

## 6. Control Characteristics
*   **Horizon:** 16 steps (Waypoints).
*   **Predictive Latency:** The model predicts the next **160-320ms** of movement (assuming 50-100Hz execution).
*   **Interface:** Task-space relative deltas (ΔEEF) -> Compatible with IKPy + URDF.
