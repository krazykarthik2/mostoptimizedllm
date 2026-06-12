# Mini π0.5 / SmolVLA for a 4-DOF Robot: Project Master Plan

## 0. Fixed Constraints
* **Dataset:** BridgeData V2 (Raw data allowed).
* **Preprocessing:** Preprocessing (Image-to-Embedding, Tokenization, Action-to-Trajectory) is now enabled as part of the pipeline.
* **Input State:** Current end-effector (EEF) state [x, y, z, g].
* **Output:** Future relative EEF trajectories (16 waypoints).
* **Robot:** 4-DOF single-gripper robot (URDF + IK).
* **Hardware:** 2× NVIDIA L4 GPUs (Note: Multi-GPU currently restricted by NCCL/Driver mismatch; fallback to single-GPU is implemented).
* **Precision:** BF16 (Optimized for L4).
* **Optimizer:** Muon (with Fast Inverse Square Root approximation).

## 1. Vision Pipeline
* **Encoder:** SigLIP Base (Frozen).
* **Offline Preprocessing:** Image -> SigLIP -> 768-dim embedding -> Save as Parquet.
* **Tokens:** 768-dim embedding projected to 8 vision tokens.

## 2. Model Architecture (SmolVLA-inspired)
* **Backbone:** SmolLM2-360M (24 layers).
* **Freezing:** Layers 0–19 frozen. Layers 20–23 + LoRA + Head are trained.
* **LoRA:** Rank 16, Alpha 32, Dropout 0.05 on Attention Q and V.
* **Token Sequence:** `[BOS] [VISION TOKENS] [STATE TOKENS] [LANGUAGE TOKENS]`
    * Vision: 8 tokens.
    * State: 2 tokens (Projected from 4-dim EEF).
    * Language: Variable length.
* **Output Head:** MLP (768 -> 64) predicting 16 × [Δx, Δy, Δz, Δg].

## 3. Training Details
* **Loss:** MSE (Trajectory) + BCE (Gripper).
* **Optimizer (Muon):** LR 1e-3, WD 0.01, Warmup 3%, Grad Clip 1.0.
* **Batch Size:** 128.
* **Precision:** FP8 via `transformer_engine` or similar.

## 4. Execution Pipeline
* **Inference:** Camera -> SigLIP -> SmolVLA -> 16-step EEF Trajectory.
* **IK:** IKPy + URDF to convert EEF waypoints to Joint Angles.
* **Simulation:** MuJoCo for validation and visualization.

## 5. Directory Structure
```text
robotmodel/
├── data/               # Precomputed embeddings (Parquet)
├── models/             # SmolLM2 backbone and trained LoRA weights
├── src/
│   ├── preprocess.py   # SigLIP embedding generation
│   ├── model.py        # SmolVLA implementation
│   ├── train.py        # Training script (Multi-GPU, FP8, Muon)
│   ├── ik.py           # IKPy + URDF integration
│   └── sim/            # MuJoCo simulation environment
├── urdf/               # Robot URDF files
├── scripts/            # Utility scripts for training/eval
└── GEMINI.md           # This file
```
