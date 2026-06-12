import torch
import numpy as np
import pandas as pd
import imageio
import json
from transformers import SiglipModel, SiglipProcessor, AutoTokenizer
from tqdm import tqdm
import os
import glob
from accelerate import Accelerator

def process_dataset(data_root, output_dir, limit_episodes=None):
    accelerator = Accelerator()
    device = accelerator.device
    
    tasks_path = os.path.join(data_root, "meta/tasks.jsonl")
    info_path = os.path.join(data_root, "meta/info.json")
    
    print(f"Loading tasks from {tasks_path}")
    tasks = {}
    if os.path.exists(tasks_path):
        with open(tasks_path, 'r') as f:
            for line in f:
                t = json.loads(line)
                tasks[t['task_index']] = t['task']
    else:
        print("Warning: tasks.jsonl not found. Using empty tasks.")

    print("Loading models...")
    vision_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files in data/chunk-*/
    parquet_files = sorted(glob.glob(os.path.join(data_root, "data/chunk-*/episode_*.parquet")))
    
    if not parquet_files:
        # Fallback search for other structures
        parquet_files = sorted(glob.glob(os.path.join(data_root, "**", "episode_*.parquet"), recursive=True))

    if limit_episodes:
        parquet_files = parquet_files[:limit_episodes]
        
    print(f"Found {len(parquet_files)} episodes to process.")

    # Pre-index video files for faster lookup
    print("Indexing video files...")
    video_map = {}
    for vp in glob.glob(os.path.join(data_root, "videos", "**", "*.mp4"), recursive=True):
        v_name = os.path.basename(vp).replace(".mp4", "")
        # Store all occurrences; prioritize the one in the same chunk
        if v_name not in video_map:
            video_map[v_name] = []
        video_map[v_name].append(vp)
    
    missing_count = 0
    for parquet_path in tqdm(parquet_files, desc="Processing Episodes"):
        episode_id = os.path.basename(parquet_path).replace(".parquet", "")
        # Extract chunk_id from path
        path_parts = parquet_path.split(os.sep)
        chunk_id = "unknown"
        for part in reversed(path_parts):
            if "chunk-" in part:
                chunk_id = part
                break
        
        # Determine output path
        episode_output_path = os.path.join(output_dir, f"{chunk_id}_{episode_id}.parquet")
        if os.path.exists(episode_output_path):
            continue
            
        # Robust video lookup
        video_path = None
        if episode_id in video_map:
            # Try to find video in the same chunk
            for vp in video_map[episode_id]:
                if chunk_id in vp:
                    video_path = vp
                    break
            # Fallback to the first found video for this episode ID
            if not video_path:
                video_path = video_map[episode_id][0]
        
        if not video_path:
            missing_count += 1
            if missing_count <= 5: # Only warn for the first 5
                print(f"Warning: Video not found for {parquet_path}. Skipping.")
            elif missing_count == 6:
                print("Further missing video warnings suppressed...")
            continue
        
        try:
            df = pd.read_parquet(parquet_path)
            reader = imageio.get_reader(video_path)
            
            processed_data = []
            frames_iter = reader.iter_data()
            
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    frame = next(frames_iter)
                except StopIteration:
                    break
                    
                inputs = processor(images=[frame], return_tensors="pt").to(device)
                with torch.no_grad():
                    vision_emb = vision_model.get_image_features(**inputs).cpu().numpy().flatten()
                    
                instruction = tasks.get(row['task_index'], "unknown task")
                input_ids = tokenizer(instruction, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.numpy().flatten()
                
                # LeRobot State is 7D: [x, y, z, roll, pitch, yaw, gripper]
                # We need 4D for our 4-DOF robot: [x, y, z, gripper]
                full_state = np.array(row['observation.state'], dtype=np.float32)
                current_eef = np.array([full_state[0], full_state[1], full_state[2], full_state[6]], dtype=np.float32) 
                
                # Action Trajectory (next 16 steps)
                future_actions = df.iloc[i:i+16]['action'].tolist()
                traj = np.zeros((16, 4), dtype=np.float32)
                
                # Normalization stats for position deltas [x, y, z]
                # Calculated from BridgeData V2 subset
                ACTION_MEAN = np.array([0.0026, -0.0042, -0.0018], dtype=np.float32)
                ACTION_STD  = np.array([0.0085, 0.0112, 0.0168], dtype=np.float32)
                
                for j, act in enumerate(future_actions):
                    # act is [x, y, z, roll, pitch, yaw, gripper]
                    # Z-score normalize x,y,z
                    norm_pos = (np.array(act[:3], dtype=np.float32) - ACTION_MEAN) / ACTION_STD
                    traj[j] = np.array([
                        norm_pos[0],
                        norm_pos[1],
                        norm_pos[2],
                        act[6] # gripper remains 0-1
                    ], dtype=np.float32)
                if len(future_actions) < 16:
                    # Pad with zero deltas (stay in place)
                    for j in range(len(future_actions), 16):
                        traj[j] = np.zeros(4, dtype=np.float32)
                        
                processed_data.append({
                    "vision_embedding": vision_emb.astype(np.float32).tolist(),
                    "current_eef": current_eef.astype(np.float32).tolist(),
                    "input_ids": input_ids.astype(np.int64).tolist(),
                    "future_trajectory": traj.flatten().astype(np.float32).tolist()
                })
            
            if processed_data:
                df_out = pd.DataFrame(processed_data)
                df_out.to_parquet(episode_output_path)
                
        except Exception as e:
            print(f"Error processing {parquet_path}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/jupyter-238w1a5447/bridge_v2_data")
    parser.add_argument("--output_dir", type=str, default="/home/jupyter-238w1a5447/robotmodel/data/processed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    process_dataset(args.data_root, args.output_dir, limit_episodes=args.limit)
