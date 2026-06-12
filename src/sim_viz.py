import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import pandas as pd
import os
import random
from src.model import SmolVLA
from src.ik import RobotIK
from transformers import AutoTokenizer

class StudioSim:
    def __init__(self, urdf_path, model_path=None):
        self.urdf_path = os.path.abspath(urdf_path)
        # Connect headlessly
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Color mapping for randomization
        self.color_map = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1],
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
            "purple": [1, 0, 1, 1],
            "cyan": [0, 1, 1, 1]
        }
        
        # Initial Environment Setup
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Table (Static)
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.3]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.3], rgbaColor=[0.8, 0.8, 0.8, 1]),
            basePosition=[0.6, 0, 0.3]
        )
        
        # Robot Pedestal (Static)
        self.pedestal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2], rgbaColor=[0.2, 0.2, 0.2, 1]),
            basePosition=[0, 0, 0.2]
        )
        
        # Load Robot
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0.4], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Joint Indices for KUKA KR6 + Parallel Gripper
        self.arm_joints = [0, 1, 2, 3, 4, 5]
        self.gripper_joints = [7, 8] # Indices 7 and 8 based on URDF structure
        self.ee_link_idx = 6 # gripper_base
        
        # Dynamic Object Placeholders
        self.block_id = None
        self.bucket_id = None
        
        # Load Models
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        from transformers import SiglipModel, SiglipProcessor
        print("Loading SigLIP for real-time observations...")
        self.siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").eval()
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.siglip_model.to(self.device)

        self.vla_model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading VLA model from {model_path}...")
            self.vla_model = SmolVLA()
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state"]
            # Fix state dict mapping for compiled models
            new_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
            self.vla_model.load_state_dict(new_state_dict)
            self.vla_model.eval()
            self.vla_model.to(self.device)
            
        self.ik_solver = RobotIK(urdf_path)

    def randomize_environment(self):
        """Randomizes colors, positions, and lighting for a new task."""
        # 1. Randomize Colors
        b_color = random.choice(list(self.color_map.keys()))
        t_color = random.choice([c for c in self.color_map.keys() if c != b_color])
        
        # 2. Randomize Positions
        block_pos = [random.uniform(0.45, 0.55), random.uniform(-0.15, 0.15), 0.62]
        bucket_pos = [random.uniform(0.65, 0.75), random.uniform(-0.25, 0.25), 0.61]
        
        # Clean up old objects
        if self.block_id is not None: p.removeBody(self.block_id)
        if self.bucket_id is not None: p.removeBody(self.bucket_id)
            
        # Create randomized cube
        self.block_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=self.color_map[b_color]),
            basePosition=block_pos
        )
        p.changeDynamics(self.block_id, -1, lateralFriction=2.0, rollingFriction=0.1)
        
        # Create randomized target bucket (tray)
        self.bucket_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.02]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.02], rgbaColor=self.color_map[t_color]),
            basePosition=bucket_pos
        )
        
        instruction = f"put the {b_color} cube in the {t_color} bucket"
        return instruction, block_pos

    def get_observation(self):
        """Captures a 224x224 image from the observation camera and returns SigLIP embedding."""
        obs_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, -0.5, 0.9],
            cameraTargetPosition=[0.6, 0, 0.6],
            cameraUpVector=[0, 0, 1]
        )
        obs_proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=10.0)
        
        (_, _, rgb, _, _) = p.getCameraImage(224, 224, obs_view_matrix, obs_proj_matrix, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (224, 224, 4))[:, :, :3]
        
        inputs = self.siglip_processor(images=[rgb], return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_emb = self.siglip_model.get_image_features(**inputs)
        return vision_emb

    def run_simulation(self, output_video="viz/final_trajectory.mp4"):
        # Initialize
        instruction, block_start_pos = self.randomize_environment()
        print(f"Executing Randomized Task: {instruction}")
        
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        width, height = 640, 480
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
        
        # Studio Recording Camera (Fixed Angle)
        rec_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.5, -1.0, 1.2],
            cameraTargetPosition=[0.5, 0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
        rec_proj_matrix = p.computeProjectionMatrixFOV(fov=45, aspect=float(width)/height, nearVal=0.1, farVal=100.0)

        # Tokenize Instruction
        input_ids = self.tokenizer(instruction, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.to(self.device)

        # Initial Pose: Position gripper just above the block
        ready_pos = [block_start_pos[0], block_start_pos[1], block_start_pos[2] + 0.1]
        initial_joint_angles = p.calculateInverseKinematics(self.robot_id, self.ee_link_idx, ready_pos)
        for j in range(len(initial_joint_angles)):
            p.resetJointState(self.robot_id, j, initial_joint_angles[j])
            
        # Initial tracking state for model
        curr_eef_pos = np.array(ready_pos)
        ACTION_SCALE = 50.0

        # Multi-cycle Execution
        for cycle in range(6): # Run 6 cycles of 16 steps each
            print(f"  Cycle {cycle+1}/6: Capturing observation and predicting...", end="", flush=True)
            # 1. Capture observation for the model
            vision_emb = self.get_observation()
            
            # 2. Capture current state [x, y, z, g]
            # Get actual gripper position from physics
            ee_state = p.getLinkState(self.robot_id, self.ee_link_idx)
            actual_pos = ee_state[4] # worldLinkFramePosition
            
            state_tensor = torch.tensor([[actual_pos[0], actual_pos[1], actual_pos[2], 1.0]], dtype=torch.float32).to(self.device)

            # 3. Predict next trajectory chunk
            with torch.no_grad():
                pred_traj = self.vla_model(vision_emb, state_tensor, input_ids)
                trajectory = pred_traj.view(16, 4).cpu().numpy()

            avg_delta = np.abs(trajectory[:, :3]).mean()
            print(f" Done. (Avg Pred Delta: {avg_delta:.4f})", flush=True)

            # 4. Physical Execution of the chunk
            for i in range(len(trajectory)):
                if i % 4 == 0:
                    print(f"    Step {i+1}/16...", end="\r", flush=True)
                # Calculate absolute target from relative delta
                delta = trajectory[i][:3] / ACTION_SCALE
                gripper_val = trajectory[i][3] # 0 closed, 1 open
                
                target_pos = actual_pos + delta
                
                # Update IK for target position
                ik_angles = p.calculateInverseKinematics(self.robot_id, self.ee_link_idx, target_pos)
                
                # Gripper travel: 0 (closed) to 0.025 (open)
                # Map model 0.0-1.0 to prismatic travel
                gripper_pos = gripper_val * 0.025
                
                # Physics steps per waypoint
                for _ in range(5):
                    # Set Motor Targets
                    for j_idx, angle in enumerate(ik_angles):
                        if j_idx < 6: # Arm Joints
                            p.setJointMotorControl2(self.robot_id, j_idx, p.POSITION_CONTROL, targetPosition=angle, force=300)
                    
                    # Gripper control
                    for g_idx in self.gripper_joints:
                        p.setJointMotorControl2(self.robot_id, g_idx, p.POSITION_CONTROL, targetPosition=gripper_pos, force=50)
                        
                    p.stepSimulation()
                    
                    # Record a frame from the Studio Camera
                    (_, _, px, _, _) = p.getCameraImage(width, height, rec_view_matrix, rec_proj_matrix, renderer=p.ER_TINY_RENDERER)
                    rgb_array = cv2.cvtColor(np.reshape(px, (height, width, 4))[:, :, :3], cv2.COLOR_RGB2BGR)
                    out.write(rgb_array)
                
                # Update current actual position for the next delta calculation
                actual_pos = p.getLinkState(self.robot_id, self.ee_link_idx)[4]
                
        out.release()
        print(f"Randomized Simulation saved to {output_video}")

if __name__ == "__main__":
    # Ensure src is in path if running directly
    import sys
    sys.path.append(os.getcwd())
    
    sim = StudioSim(
        "urdf/kuka_kr6.urdf", 
        "robotmodel/models/checkpoints/latest.pt"
    )
    
    num_videos = 5
    for i in range(num_videos):
        video_path = f"viz/sim_video_{i+1}.mp4"
        print(f"\\n--- Generating Video {i+1}/{num_videos} ---")
        sim.run_simulation(video_path)
