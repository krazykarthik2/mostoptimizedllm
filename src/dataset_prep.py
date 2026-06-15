import torch
import numpy as np
import pandas as pd
import imageio
import json
from tqdm import tqdm
import os
import glob
from accelerate import Accelerator
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from src.canonical import normalize_state, normalize_action

def process_dataset(data_root, output_dir, limit_episodes=None):
    accelerator = Accelerator()
    device = accelerator.device

    tasks_path = os.path.join(data_root, "meta/tasks.jsonl")
    print(f"Loading tasks from {tasks_path}")
    tasks = {}
    if os.path.exists(tasks_path):
        with open(tasks_path, 'r') as f:
            for line in f:
                t = json.loads(line)
                tasks[t['task_index']] = t['task']

    print("Loading models...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    vlm_model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.bfloat16
    ).to(device).eval()

    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(data_root, "data/chunk-*/episode_*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(data_root, "**", "episode_*.parquet"), recursive=True))

    if limit_episodes:
        parquet_files = parquet_files[:limit_episodes]
        
    print("Indexing video files...")
    video_map = {}
    for vp in glob.glob(os.path.join(data_root, "videos", "**", "*.mp4"), recursive=True):
        v_name = os.path.basename(vp).replace(".mp4", "")
        if v_name not in video_map: video_map[v_name] = []
        video_map[v_name].append(vp)
    
    for parquet_path in tqdm(parquet_files, desc="Processing Episodes"):
        episode_id = os.path.basename(parquet_path).replace(".parquet", "")
        path_parts = parquet_path.split(os.sep)
        chunk_id = "unknown"
        for part in reversed(path_parts):
            if "chunk-" in part:
                chunk_id = part
                break
        
        episode_output_path = os.path.join(output_dir, f"{chunk_id}_{episode_id}.parquet")
        if os.path.exists(episode_output_path): continue
            
        video_path = None
        if episode_id in video_map:
            for vp in video_map[episode_id]:
                if chunk_id in vp:
                    video_path = vp
                    break
            if not video_path: video_path = video_map[episode_id][0]
        
        if not video_path: continue
        
        try:
            df = pd.read_parquet(parquet_path)
            reader = imageio.get_reader(video_path)
            processed_data = []
            frames_iter = reader.iter_data()
            
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    frame = next(frames_iter)
                except StopIteration: break
                    
                inputs = processor(images=[frame], size={"longest_edge": 512}, return_tensors="pt").to(device, dtype=torch.bfloat16)
                with torch.no_grad():
                    pixel_values = inputs.pixel_values.view(-1, 3, 512, 512)
                    vision_outputs = vlm_model.model.vision_model(pixel_values=pixel_values)
                    v_tokens = vlm_model.model.connector(vision_outputs.last_hidden_state)
                    vision_emb = v_tokens.cpu().to(torch.float32).numpy().flatten()
                
                # USE CANONICAL NORMALIZATION
                full_state = np.array(row['observation.state'], dtype=np.float32)
                current_eef = normalize_state(full_state[:3], full_state[6])
                
                instruction = tasks.get(row['task_index'], "unknown task")
                input_ids = tokenizer(instruction, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.numpy().flatten()

                future_actions = df.iloc[i:i+16]['action'].tolist()
                traj = np.zeros((16, 4), dtype=np.float32)
                
                prev_pos = full_state[:3]
                for j, act in enumerate(future_actions):
                    raw_pos = np.array(act[:3], dtype=np.float32)
                    norm_act = normalize_action(raw_pos - prev_pos, act[6])
                    traj[j] = norm_act
                    prev_pos = raw_pos

                if len(future_actions) < 16:
                    for j in range(len(future_actions), 16):
                        traj[j] = np.array([-1, -1, -1, 1], dtype=np.float32) 
                        
                processed_data.append({
                    "vision_embedding": vision_emb.astype(np.float32).tolist(),
                    "current_eef": current_eef.tolist(),
                    "input_ids": input_ids.tolist(),
                    "future_trajectory": traj.flatten().tolist()
                })
            
            if processed_data:
                pd.DataFrame(processed_data).to_parquet(episode_output_path)
        except Exception: continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/jupyter-238w1a5447/bridge_v2_data")
    parser.add_argument("--output_dir", type=str, default="/home/jupyter-238w1a5447/robotmodel/data/processed_canonical")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    process_dataset(args.data_root, args.output_dir, limit_episodes=args.limit)
