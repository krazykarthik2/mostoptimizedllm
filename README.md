# SmolVLA for 4-DOF Robot

This project implements a lightweight Vision-Language-Action (VLA) model based on the SmolLM2-360M architecture, optimized for a 4-DOF robot using BridgeData V2.

## Key Features
- **Backbone:** SmolLM2-360M (24 layers used).
- **Vision:** Frozen SigLIP-Base embeddings (768-dim) projected to 8 tokens.
- **Action:** 16-step relative End-Effector (EEF) trajectory prediction.
- **Optimizer:** Muon (Momentum Orthogonalized by Newton-Schulz).
- **Precision:** FP8 (supported via training loop).
- **Hardware:** Designed for 2x NVIDIA L4 GPUs.

## Getting Started

### 1. Preprocess Data
Generate SigLIP embeddings for BridgeData V2:
```bash
python3 src/dataset_prep.py
```

### 2. Train
Launch multi-GPU training:
```bash
./scripts/run_train.sh
```

### 3. Inference & IK
Convert predicted trajectories to joint angles:
```python
from src.ik import RobotIK
ik = RobotIK("urdf/robot.urdf")
joint_angles = ik.solve_ik(target_pos)
```

## Directory Structure
- `src/`: Core implementation (Model, Optimizer, Train, IK).
- `urdf/`: Robot URDF definitions.
- `data/`: Precomputed embeddings (Parquet).
- `scripts/`: Utility scripts.
- `GEMINI.md`: Master project plan and constraints.
# robotmodel
