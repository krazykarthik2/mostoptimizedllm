import os
import torch
import numpy as np
import pybullet as p
import pybullet_data
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoProcessor, AutoModelForVision2Seq
from src.model import SmolVLA
from src.canonical import normalize_state, denormalize_action
import imageio

class FirstGuessSim:
    def __init__(self, model_path="robotmodel/models/checkpoints/latest.pt"):
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.color_map = {"red":[1,0,0,1],"green":[0,1,0,1],"blue":[0,0,1,1],"yellow":[1,1,0,1],"purple":[1,0,1,1],"cyan":[0,1,1,1]}
        p.setGravity(0, 0, -9.81); p.loadURDF("plane.urdf")
        self.mat_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.001]), baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.001], rgbaColor=[0.95,0.95,0.95,1]), basePosition=[0.4,0,0.001])
        self.gripper_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02,0.04,0.02]), baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02,0.04,0.02], rgbaColor=[0.2,0.2,0.2,1]), basePosition=[0.3,0,0.2])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.vlm_base = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.bfloat16).to(self.device).eval()
        self.vla_model = SmolVLA().to(self.device)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            sd = ckpt["model_state"]; nsd = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
            self.vla_model.load_state_dict(nsd)
        self.vla_model.eval()

    def generate_first_guess(self, output_path="viz/first_guess.png"):
        b, t = random.sample(list(self.color_map.keys()), 2)
        bp, up = [random.uniform(0.3, 0.45), random.uniform(-0.15, 0.15), 0.02], [random.uniform(0.5, 0.65), random.uniform(-0.2, 0.2), 0.01]
        p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015,0.015,0.015], rgbaColor=self.color_map[b]), basePosition=bp)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06,0.06,0.01], rgbaColor=self.color_map[t]), basePosition=up)
        inst = f"put the {b} cube in the {t} bucket"
        actual_pos = np.array([random.uniform(0.25, 0.45), random.uniform(-0.15, 0.15), 0.25])
        p.resetBasePositionAndOrientation(self.gripper_id, actual_pos, [0, 0, 0, 1])
        vm, pm = p.computeViewMatrix([-0.1, 0.4, 0.4], [0.4, 0.0, 0.1], [0, 0, 1]), p.computeProjectionMatrixFOV(50, 1.0, 0.1, 10.0)
        (_, _, rgb, _, _) = p.getCameraImage(224, 224, vm, pm, renderer=p.ER_TINY_RENDERER)
        inputs = self.processor(images=[np.reshape(rgb, (224, 224, 4))[:, :, :3]], size={"longest_edge": 512}, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            px = inputs.pixel_values.view(-1, 3, 512, 512); vo = self.vlm_base.model.vision_model(pixel_values=px)
            vt = self.vlm_base.model.connector(vo.last_hidden_state); ve = vt.cpu().to(torch.float32).numpy().flatten()
            ve = torch.tensor(ve).unsqueeze(0).to(self.device)
            ns = normalize_state(actual_pos, 1.0); st = torch.tensor([ns], dtype=torch.float32).to(self.device)
            ids = self.processor.tokenizer(inst, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.to(self.device)
            pred = self.vla_model.predict_action(ve, st, ids, num_steps=64).view(16, 4).cpu().numpy()
        
        path = [actual_pos]
        curr = actual_pos.copy()
        for i in range(16):
            d, _ = denormalize_action(pred[i])
            curr += d; path.append(curr.copy())
        path = np.stack(path)
        
        # Gripper states
        grippers = pred[:, 3]
        
        fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
        ax.plot(path[:,0], path[:,1], path[:,2], 'r-', alpha=1.0, label='Guess Path')
        
        for i in range(len(grippers)):
            m = 'o' if grippers[i] > 0 else 'x'
            ax.scatter(path[i+1,0], path[i+1,1], path[i+1,2], color='red', marker=m, s=60, alpha=1.0)
        
        ax.scatter(bp[0], bp[1], bp[2], color=self.color_map[b], s=200, alpha=1.0, label='Cube')
        ax.scatter(up[0], up[1], up[2], color=self.color_map[t], s=400, alpha=1.0, label='Bucket')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', label='Guess Path'),
            Line2D([0], [0], marker='o', color='w', label='Open (Predicted)', markerfacecolor='black', markersize=10),
            Line2D([0], [0], marker='x', color='w', label='Closed (Predicted)', markeredgecolor='black', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Cube/Bucket', markerfacecolor='gray', markersize=10)
        ]
        
        ax.set_title(inst); ax.legend(handles=legend_elements); os.makedirs("viz", exist_ok=True); plt.savefig(output_path)
        
        # Capture Rotating Video
        frames = []
        print("Generating rotating 3D video...")
        for angle in range(0, 360, 5):
            ax.view_init(elev=20, azim=angle)
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            frames.append(image)
        
        video_path = output_path.replace(".png", ".mp4")
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Video saved to {video_path}")
        
        p.disconnect()

if __name__ == "__main__":
    FirstGuessSim().generate_first_guess()
