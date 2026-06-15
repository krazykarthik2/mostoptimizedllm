import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import pandas as pd
import os
import random
from src.model import SmolVLA
from transformers import AutoTokenizer
from src.canonical import normalize_state, denormalize_action

class FloatingSim:
    def __init__(self, model_path=None):
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.color_map = {"red":[1,0,0,1],"green":[0,1,0,1],"blue":[0,0,1,1],"yellow":[1,1,0,1],"purple":[1,0,1,1],"cyan":[0,1,1,1]}
        p.setGravity(0, 0, -9.81); p.loadURDF("plane.urdf")
        self.mat_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.001]), baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.001], rgbaColor=[0.95,0.95,0.95,1]), basePosition=[0.4,0,0.001])
        self.gripper_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02,0.04,0.02]), baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02,0.04,0.02], rgbaColor=[0.2,0.2,0.2,1]), basePosition=[0.3,0,0.2])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoProcessor, AutoModelForVision2Seq
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.tokenizer = self.processor.tokenizer
        self.vlm_base = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.bfloat16).to(self.device).eval()
        self.vla_model = None
        if model_path and os.path.exists(model_path):
            self.vla_model = SmolVLA().to(self.device)
            ckpt = torch.load(model_path, map_location="cpu")
            sd = ckpt["model_state"]; nsd = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
            self.vla_model.load_state_dict(nsd); self.vla_model.eval()

    def randomize_environment(self):
        b, t = random.sample(list(self.color_map.keys()), 2)
        bp, up = [random.uniform(0.3, 0.45), random.uniform(-0.15, 0.15), 0.02], [random.uniform(0.5, 0.65), random.uniform(-0.2, 0.2), 0.01]
        self.block_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015,0.015,0.015]), baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015,0.015,0.015], rgbaColor=self.color_map[b]), basePosition=bp)
        self.bucket_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06,0.06,0.01]), baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06,0.06,0.01], rgbaColor=self.color_map[t]), basePosition=up)
        return f"put the {b} cube in the {t} bucket", bp

    def get_observation(self):
        vm = p.computeViewMatrix([-0.1, 0.4, 0.4], [0.4, 0.0, 0.1], [0, 0, 1])
        pm = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 10.0)
        (_, _, rgb, _, _) = p.getCameraImage(224, 224, vm, pm, renderer=p.ER_TINY_RENDERER)
        inputs = self.processor(images=[np.reshape(rgb, (224, 224, 4))[:, :, :3]], size={"longest_edge": 512}, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            px = inputs.pixel_values.view(-1, 3, 512, 512)
            vo = self.vlm_base.model.vision_model(pixel_values=px)
            vt = self.vlm_base.model.connector(vo.last_hidden_state)
            ve = vt.cpu().to(torch.float32).numpy().flatten()
        return torch.tensor(ve).unsqueeze(0).to(self.device)

    def run_simulation(self, output_video):
        if not self.vla_model: return
        inst, _ = self.randomize_environment(); width, height = 640, 480
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
        rvm = p.computeViewMatrix([1.0, 0.8, 1.0], [0.4, 0, 0.1], [0, 0, 1])
        rpm = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100.0)
        ids = self.tokenizer(inst, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.to(self.device)
        actual_pos = np.array([random.uniform(0.25, 0.45), random.uniform(-0.15, 0.15), 0.25])
        p.resetBasePositionAndOrientation(self.gripper_id, actual_pos, [0, 0, 0, 1])
        gc = None
        for cycle in range(12):
            ve = self.get_observation(); ns = normalize_state(actual_pos, 1.0)
            st = torch.tensor([ns], dtype=torch.float32).to(self.device)
            with torch.no_grad(): traj = self.vla_model.predict_action(ve, st, ids, num_steps=32).view(16, 4).cpu().numpy()
            for i in range(16):
                phys_d, gv = denormalize_action(traj[i]); actual_pos += phys_d
                p.resetBasePositionAndOrientation(self.gripper_id, actual_pos, [0, 0, 0, 1])
                if gv < 0.0 and gc is None:
                    bp = p.getBasePositionAndOrientation(self.block_id)[0]
                    if np.linalg.norm(np.array(bp) - actual_pos) < 0.05:
                        gc = p.createConstraint(self.gripper_id, -1, self.block_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], np.array(bp)-actual_pos)
                elif gv >= 0.0 and gc is not None: p.removeConstraint(gc); gc = None
                for _ in range(48): p.stepSimulation()
                (_, _, px, _, _) = p.getCameraImage(width, height, rvm, rpm, renderer=p.ER_TINY_RENDERER)
                out.write(cv2.cvtColor(np.reshape(px, (height, width, 4))[:, :, :3], cv2.COLOR_RGB2BGR))
        out.release()

if __name__ == "__main__":
    import sys; sys.path.append(os.getcwd())
    sim = FloatingSim("robotmodel/models/checkpoints/latest.pt")
    for i in range(5): sim.run_simulation(f"viz/sim_video_{i+1}.mp4")
