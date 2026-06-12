import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from src.model import SmolVLA
from src.muon import MuonWithAuxAdam
import pandas as pd
import numpy as np
import hashlib
import shutil
import cv2
import imageio
import mujoco
import glob
import matplotlib.pyplot as plt

def get_architecture_hash(model):
    """Generates a hash based on the model architecture and hyperparameters."""
    model_str = str(model)
    return hashlib.sha256(model_str.encode()).hexdigest()[:12]

class BridgeEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        if os.path.isdir(data_path):
            files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            print(f"Loading dataset from {len(files)} files in {data_path}")
            dfs = []
            for f in files:
                dfs.append(pd.read_parquet(f))
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.read_parquet(data_path)
        print(f"Dataset loaded with {len(self.df)} samples.")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            return {
                "vision": torch.tensor(np.array(row['vision_embedding'].tolist() if hasattr(row['vision_embedding'], 'tolist') else row['vision_embedding'], dtype=np.float32)),
                "state": torch.tensor(np.array(row['current_eef'].tolist() if hasattr(row['current_eef'], 'tolist') else row['current_eef'], dtype=np.float32)),
                "input_ids": torch.tensor(np.array(row['input_ids'].tolist() if hasattr(row['input_ids'], 'tolist') else row['input_ids'], dtype=np.int64)),
                "target": torch.tensor(np.array(row['future_trajectory'].tolist() if hasattr(row['future_trajectory'], 'tolist') else row['future_trajectory'], dtype=np.float32)).view(-1)
            }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            print(f"future_trajectory type: {type(row['future_trajectory'])}")
            if hasattr(row['future_trajectory'], 'shape'):
                print(f"future_trajectory shape: {row['future_trajectory'].shape}")
            raise e

def loss_fn(pred, target):
    # Ensure high precision for loss calculation, especially for small deltas
    pred = pred.view(-1, 16, 4).float()
    target = target.view(-1, 16, 4).float()
    
    # Scale position loss. With Z-score normalization, targets are unit variance,
    # so a weight of 1.0-10.0 is usually sufficient. 
    pos_loss = F.mse_loss(pred[:, :, :3], target[:, :, :3]) * 10.0
    
    # Gripper is a binary state, BCE is appropriate
    gripper_pred = torch.sigmoid(pred[:, :, 3])
    gripper_target = torch.clamp(target[:, :, 3], 0.0, 1.0)
    gripper_loss = F.binary_cross_entropy(gripper_pred, gripper_target)
    
    return pos_loss + gripper_loss

def visualize_trajectory(model, batch, device, step, output_dir="viz"):
    """Generates a visualization plot comparing predicted and target trajectories."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use uncompiled model if available to avoid dynamo errors during eval
    if hasattr(model, '_orig_mod'):
        eval_model = model._orig_mod
    else:
        eval_model = model
        
    eval_model.eval()
    with torch.no_grad():
        vision = batch['vision'][0:1].to(device)
        state = batch['state'][0:1].to(device)
        input_ids = batch['input_ids'][0:1].to(device)
        
        pred = eval_model(vision, state, input_ids)
        pred = pred.view(16, 4).cpu().numpy()
        target = batch['target'][0].view(16, 4).cpu().numpy()
        start_pos = state[0, :3].cpu().numpy()

    # De-normalize (Z-score stats)
    ACTION_MEAN = np.array([0.0026, -0.0042, -0.0018], dtype=np.float32)
    ACTION_STD  = np.array([0.0085, 0.0112, 0.0168], dtype=np.float32)
    
    pred_pos = (pred[:, :3] * ACTION_STD) + ACTION_MEAN
    target_pos = (target[:, :3] * ACTION_STD) + ACTION_MEAN

    # Integrate deltas
    px = start_pos[0] + np.cumsum(pred_pos[:, 0])
    py = start_pos[1] + np.cumsum(pred_pos[:, 1])
    pz = start_pos[2] + np.cumsum(pred_pos[:, 2])
    
    tx = start_pos[0] + np.cumsum(target_pos[:, 0])
    ty = start_pos[1] + np.cumsum(target_pos[:, 1])
    tz = start_pos[2] + np.cumsum(target_pos[:, 2])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tx, ty, tz, 'b--x', label='Ground Truth', alpha=0.6)
    ax.plot(px, py, pz, 'r-o', label='Predicted')
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=100)
    ax.set_title(f'Step {step} Trajectory')
    ax.legend()
    
    plt.savefig(f"{output_dir}/step_{step}.png")
    plt.close(fig)
    print(f"Visualization saved to {output_dir}/step_{step}.png")
    model.train()

import time
from tqdm import tqdm

# Fix for torch.compile issues in some environments
if hasattr(torch, "_dynamo"):
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

def train(overfit=False):
    # Performance Optimization: Enable TensorFloat32 for matmuls on Ampere/Lovelace
    torch.set_float32_matmul_precision('high')
    
    # Use BF16 precision for L4 GPUs (Ada Lovelace)
    accelerator = Accelerator(mixed_precision="bf16") 
    device = accelerator.device
    
    model = SmolVLA()
    
    if overfit:
        print("!!! RUNNING IN OVERFIT MODE (Single Sample) !!!")
        # Compile is slower for single sample tests, skip it
    elif hasattr(torch, "compile"):
        print("Compiling model for maximum throughput...")
        model = torch.compile(model)
    
    arch_version = get_architecture_hash(model)
    checkpoint_dir = "robotmodel/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    start_step = 0
    
    # Versioning & Checkpoint Management (Skip for overfit)
    if not overfit and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if checkpoint.get("arch_version") == arch_version:
            print(f"Resuming from checkpoint (version {arch_version})")
            model.load_state_dict(checkpoint["model_state"])
            start_step = checkpoint["step"]
        else:
            print(f"Architecture mismatch (Old: {checkpoint.get('arch_version')}, New: {arch_version}). Deleting old checkpoints.")
            shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = MuonWithAuxAdam(model, lr=1e-3, adam_lr=3e-4)
    
    # Dataset
    data_path = "/home/jupyter-238w1a5447/robotmodel/data/processed"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        data_path = "/home/jupyter-238w1a5447/robotmodel/data/processed_bridge.parquet"
        
    train_dataset = BridgeEmbeddingDataset(data_path)
    
    if overfit:
        # Overfit on a single specific sample
        sample = train_dataset[10]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        print(f"Target values (first 4): {batch['target'][0][:4].tolist()}")
        
        model.train()
        model.to(device)
        pbar = tqdm(range(500), desc="Overfitting")
        for step in pbar:
            optimizer.zero_grad()
            outputs = model(batch['vision'], batch['state'], batch['input_ids'])
            loss = loss_fn(outputs, batch['target'])
            accelerator.backward(loss)
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        # Save one visualization of the overfit
        visualize_trajectory(model, batch, device, "overfit")
        print("Overfit complete. Result saved to viz/step_overfit.png")
        return

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    print(f"Starting training (Version: {arch_version}) on {device}")
    
    pbar = tqdm(range(start_step, 10000), desc="Training", disable=not accelerator.is_main_process)
    train_iter = iter(train_loader)
    
    model.train()
    for step in pbar:
        start_time = time.time()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        outputs = model(batch['vision'], batch['state'], batch['input_ids'])
        loss = loss_fn(outputs, batch['target'])
        accelerator.backward(loss)
        optimizer.step()
        
        # Real-time updates
        if accelerator.is_main_process:
            dt = time.time() - start_time
            samples_per_sec = 128 / dt
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "speed": f"{samples_per_sec:.1f} spl/s"
            })
        
        # Periodic Tasks
        if step % 1000 == 0 and step > start_step and accelerator.is_main_process:
            # Save Checkpoint
            checkpoint = {
                "step": step,
                "model_state": accelerator.get_state_dict(model),
                "arch_version": arch_version
            }
            torch.save(checkpoint, ckpt_path)
            print(f"\nCheckpoint saved at step {step}")
            
            # Visualization
            try:
                visualize_trajectory(model, batch, device, step)
            except Exception as e:
                print(f"Visualization failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Default is now OVERFIT mode as requested by user
    parser.add_argument("--full", action="store_true", help="Run full training on entire dataset (instead of default overfit)")
    args = parser.parse_args()
    
    # If --full is NOT passed, overfit is True
    train(overfit=not args.full)
