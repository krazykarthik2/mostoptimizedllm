import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def test(backend):
    print(f"\n--- Testing Backend: {backend} ---")
    try:
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        
        print(f"Rank {rank}/{world_size} initialized on GPU {device}")
        
        # Simple collective
        tensor = torch.ones(1).to(device) * (rank + 1)
        dist.all_reduce(tensor)
        print(f"Rank {rank} result after all_reduce: {tensor.item()}")
        
        # Simple DDP wrap
        model = nn.Linear(10, 10).to(device)
        model = DDP(model, device_ids=[device])
        print(f"Rank {rank} DDP wrap success")
        
        dist.destroy_process_group()
        print(f"--- {backend} Success ---")
    except Exception as e:
        print(f"--- {backend} Failed: {e} ---")

if __name__ == "__main__":
    # Apply workarounds
    os.environ['NCCL_IGNORE_NVML'] = '1'
    os.environ['NCCL_NVLS_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    
    backend = os.environ.get("TEST_BACKEND", "nccl")
    test(backend)
