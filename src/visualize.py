import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.model import SmolVLA
import pandas as pd

def generate_visualization(checkpoint_path, data_path, output_path="viz/final_viz.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load Model
    model = SmolVLA()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = checkpoint["model_state"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Load a sample from the dataset
    df = pd.read_parquet(data_path)
    row = df.iloc[0]
    
    vision = torch.tensor(np.array(row['vision_embedding'], dtype=np.float32)).unsqueeze(0).to(device)
    state = torch.tensor(np.array(row['current_eef'], dtype=np.float32)).unsqueeze(0).to(device)
    input_ids = torch.tensor(np.array(row['input_ids'], dtype=np.int64)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(vision, state, input_ids)
        pred = pred.view(16, 4).cpu().numpy()
        # Use np.stack to handle list of arrays correctly
        target = np.stack(row['future_trajectory'])

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    px, py, pz = pred[:, 0], pred[:, 1], pred[:, 2]
    tx, ty, tz = target[:, 0], target[:, 1], target[:, 2]
    
    # Plot predicted trajectory
    ax.plot(px, py, pz, 'r-o', label='Predicted Trajectory', markersize=4)
    # Plot target trajectory
    ax.plot(tx, ty, tz, 'b--x', label='Ground Truth', markersize=4)
    
    # Start/End points
    ax.scatter(px[0], py[0], pz[0], color='green', s=100, label='Start')
    ax.scatter(px[-1], py[-1], pz[-1], color='black', s=100, label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory Prediction (Final Loss: {checkpoint.get("loss", "N/A")})')
    ax.legend()
    
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    generate_visualization(
        "robotmodel/models/checkpoints/latest.pt",
        "data/train_embeddings.parquet"
    )
