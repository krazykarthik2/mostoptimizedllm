import torch
import torch.distributed as dist
import os

os.environ['NCCL_IGNORE_NVML'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

try:
    dist.init_process_group(backend='nccl', rank=0, world_size=1)
    print("NCCL Init Success with NCCL_IGNORE_NVML=1")
    dist.destroy_process_group()
except Exception as e:
    print(f"NCCL Init Failed: {e}")
