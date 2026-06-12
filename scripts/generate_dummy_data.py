import pandas as pd
import numpy as np
import os

def generate_dummy_data(output_path, num_samples=1000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    for i in range(num_samples):
        # Gripper target must be in [0, 1] for BCE loss
        traj = np.random.randn(16, 4).astype(np.float32)
        traj[:, 3] = np.clip(traj[:, 3], 0, 1) # Simple clip for dummy
        
        data.append({
            "vision_embedding": np.random.randn(768).astype(np.float32).tolist(),
            "current_eef": np.random.randn(4).astype(np.float32).tolist(),
            "input_ids": np.random.randint(0, 48000, (20,)).tolist(), # Safe vocab range
            "future_trajectory": traj.tolist()
        })
        
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Dummy data generated at {output_path}")

if __name__ == "__main__":
    generate_dummy_data("robotmodel/data/train_embeddings.parquet")
