import os
# CRITICAL WORKAROUND: System has NVML driver mismatch which breaks NCCL.
# We force the GLOO backend which bypasses NVML and handles distributed training over CPU/SHM.
os.environ['PYTORCH_DISTRIBUTED_BACKEND'] = 'gloo'
os.environ['ACCELERATE_TORCH_DEVICE'] = 'cuda'
os.environ['NCCL_IGNORE_NVML'] = '1'

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
import glob
import matplotlib.pyplot as plt
import json
from src.canonical import denormalize_state, denormalize_action

def get_architecture_hash(model):
    if hasattr(model, '_orig_mod'): model_str = str(model._orig_mod)
    else: model_str = str(model)
    return hashlib.sha256(model_str.encode()).hexdigest()[:12]

class BridgeEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        if os.path.isdir(data_path):
            files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            dfs = [pd.read_parquet(f) for f in files]
            self.df = pd.concat(dfs, ignore_index=True)
        else: self.df = pd.read_parquet(data_path)
        
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "vision": torch.tensor(np.array(row['vision_embedding'], dtype=np.float32)),
            "state": torch.tensor(np.array(row['current_eef'], dtype=np.float32)),
            "input_ids": torch.tensor(np.array(row['input_ids'], dtype=np.int64)),
            "target": torch.tensor(np.array(row['future_trajectory'], dtype=np.float32)).view(-1)
        }

def loss_fn(model, batch, device):
    vision = batch['vision'].to(device)
    state = batch['state'].to(device)
    input_ids = batch['input_ids'].to(device)
    x1 = batch['target'].to(device)
    batch_size = x1.shape[0]
    
    # Standard CFM Loss - Use Huber for stable but precise matching
    tau = torch.rand(batch_size, 1, device=device)
    x0 = torch.randn_like(x1)
    xt = tau * x1 + (1.0 - tau) * x0
    pred_v = model(vision, state, input_ids, noisy_actions=xt, tau=tau)
    cfm_loss = F.huber_loss(pred_v.float(), (x1 - x0).float(), delta=0.5)
    
    # Auxiliary "First Guess" Loss: Predict x1 directly from zero noise at t=0
    # Higher weight (5.0) to force perfect matching on training data.
    v0 = model(vision, state, input_ids, noisy_actions=torch.zeros_like(x1), tau=torch.zeros_like(tau))
    aux_loss = F.l1_loss(v0.float(), x1.float())
    
    return cfm_loss + 5.0 * aux_loss

def visualize_trajectory(model, batch, device, step, output_dir="viz"):
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    eval_model.eval()
    with torch.no_grad():
        vision, state, input_ids = batch['vision'][0:1].to(device), batch['state'][0:1].to(device), batch['input_ids'][0:1].to(device)
        # Use more integration steps for smoother visualization
        pred = eval_model.predict_action(vision, state, input_ids, num_steps=64).view(16, 4).cpu().numpy()
        target = batch['target'][0].view(16, 4).cpu().numpy()
        start_pos_norm = state[0].cpu().numpy()

    # USE CANONICAL DE-NORMALIZATION
    start_pos_phys, _ = denormalize_state(start_pos_norm)
    
    def get_path(deltas, start):
        path = [start]
        curr = start.copy()
        for d_norm in deltas:
            d_phys, _ = denormalize_action(d_norm)
            curr += d_phys
            path.append(curr.copy())
        return np.stack(path)

    p_path = get_path(pred, start_pos_phys)
    t_path = get_path(target, start_pos_phys)
    
    # Extract gripper states (index 3)
    p_gripper = pred[:, 3]
    t_gripper = target[:, 3]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot paths with full opacity
    ax.plot(t_path[:,0], t_path[:,1], t_path[:,2], 'b-', alpha=1.0, linewidth=1, label='GT Path')
    ax.plot(p_path[:,0], p_path[:,1], p_path[:,2], 'r-', alpha=1.0, linewidth=1, label='Pred Path')
    
    # Scatter waypoints with distinct markers for gripper state (alpha=1.0)
    # Open (> 0): 'o', Closed (<= 0): 'x'
    for i in range(len(p_gripper)):
        # GT points
        m_t = 'o' if t_gripper[i] > 0 else 'x'
        ax.scatter(t_path[i+1,0], t_path[i+1,1], t_path[i+1,2], color='blue', marker=m_t, s=50, alpha=1.0)
        
        # Pred points
        m_p = 'o' if p_gripper[i] > 0 else 'x'
        ax.scatter(p_path[i+1,0], p_path[i+1,1], p_path[i+1,2], color='red', marker=m_p, s=50, alpha=1.0)
    
    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label='GT Path'),
        Line2D([0], [0], color='red', label='Pred Path'),
        Line2D([0], [0], marker='o', color='w', label='Open Gripper', markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='x', color='w', label='Closed Gripper', markeredgecolor='black', markersize=10)
    ]
    
    ax.set_title(f'Step {step} - 3D Trajectory (o=Open, x=Closed)')
    ax.legend(handles=legend_elements)
    plt.savefig(f"{output_dir}/step_{step}.png")
    plt.close(fig)
    model.train()

def train(overfit=False):
    torch.set_float32_matmul_precision('high')
    accelerator = Accelerator(mixed_precision="bf16")
    model = SmolVLA()
    if not overfit and hasattr(torch, "compile"): 
        if accelerator.is_main_process:
            print("!!! MODE: FULL TRAINING ON ENTIRE DATASET !!!")
            print("Compiling model for maximum throughput (this may take 2-5 minutes)...", flush=True)
        model = torch.compile(model)
    
    arch_version = get_architecture_hash(model)
    ckpt_dir = "robotmodel/models/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    
    from tqdm import tqdm
    # Increase adam_lr to 3e-4 to match Muon's speed
    optimizer = MuonWithAuxAdam(model, lr=1e-3, adam_lr=3e-4)
    # Add Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.adam, T_max=20000, eta_min=1e-5)
    
    data_path = "data/processed"
    train_dataset = BridgeEmbeddingDataset(data_path)
    
    if overfit:
        batch = {k: v.unsqueeze(0).to(accelerator.device) for k, v in train_dataset[10].items()}
        # Increase capacity and steps for overfitting
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for s in range(10000):
            opt.zero_grad()
            loss = loss_fn(model, batch, accelerator.device)
            accelerator.backward(loss)
            opt.step()
            if s % 1000 == 0 and accelerator.is_main_process:
                print(f"Overfit Step {s}, Loss: {loss.item():.6f}")
                visualize_trajectory(model, batch, accelerator.device, f"overfit_{s}")
        if accelerator.is_main_process:
            visualize_trajectory(model, batch, accelerator.device, "overfit_final")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save({"model_state": accelerator.get_state_dict(model), "arch_version": arch_version}, ckpt_path)
        return

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    if accelerator.is_main_process:
        print(f"Dataset loaded with {len(train_dataset)} samples.")
        print(f"Starting training (Version: {arch_version}) on {accelerator.device}")
        
    train_iter = iter(train_loader)
    pbar = tqdm(range(20000), disable=not accelerator.is_main_process, desc="Training", mininterval=1.0)
    for step in pbar:
        try: batch = next(train_iter)
        except: train_iter = iter(train_loader); batch = next(train_iter)
        optimizer.zero_grad(); loss = loss_fn(model, batch, accelerator.device)
        accelerator.backward(loss)
        
        # Gradient clipping for stability with higher learning rates
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        if accelerator.is_main_process:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if step % 10 == 0: # Heartbeat every 10 steps
                 pbar.update(0) 
        
        if step % 1000 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                torch.save({"step": step, "model_state": accelerator.get_state_dict(model), "arch_version": arch_version}, ckpt_path)
                visualize_trajectory(model, batch, accelerator.device, step)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    train(overfit=not args.full)
