import mujoco
import numpy as np
import cv2
import torch
import pandas as pd
import os

# Set MuJoCo to use software rendering (OSMesa) or EGL for headless environments
os.environ["MUJOCO_GL"] = "osmesa" 

from src.model import SmolVLA
from src.ik import RobotIK

class MuJoCoRobotSim:
    def __init__(self, urdf_path, model_path=None):
        self.urdf_path = os.path.abspath(urdf_path)
        
        # Load URDF directly into MuJoCo
        try:
            # MuJoCo 3.0+ can load URDF directly. 
            # We might need to wrap it in a worldbody if we want a floor.
            # But let's try loading it directly first.
            self.model = mujoco.MjModel.from_xml_path(self.urdf_path)
        except Exception as e:
            print(f"Error loading URDF directly: {e}")
            # Fallback: create a very simple robot
            self.model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <worldbody>
        <light pos="0 0 3" />
        <geom type="plane" size="1 1 .01" />
        <body name="base" pos="0 0 0.1">
            <joint type="free" />
            <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1" />
        </body>
    </worldbody>
</mujoco>
""")
        
        self.data = mujoco.MjData(self.model)
        # We need an offscreen context for rendering
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Load Robot Model (trained)
        self.vla_model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading VLA model from {model_path}...")
            self.vla_model = SmolVLA()
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            self.vla_model.load_state_dict(new_state_dict)
            self.vla_model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vla_model.to(self.device)
            
        self.ik_solver = RobotIK(urdf_path)
        
    def get_trajectory(self, data_row):
        if self.vla_model is None:
            # Return ground truth if no model
            return np.array(data_row['future_trajectory']).reshape(16, 4)
            
        vision = torch.tensor(np.array(data_row['vision_embedding'], dtype=np.float32)).unsqueeze(0).to(self.device)
        state = torch.tensor(np.array(data_row['current_eef'], dtype=np.float32)).unsqueeze(0).to(self.device)
        input_ids = torch.tensor(np.array(row['input_ids'], dtype=np.int64)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.vla_model(vision, state, input_ids)
            pred = pred.view(16, 4).cpu().numpy()
        return pred

    def run_simulation(self, trajectory, output_video="viz/mujoco_simulation.mp4"):
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 20.0, (640, 480))
        
        ACTION_SCALE = 50.0
        
        # Initial joint angles
        self.data.qpos[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        # Initial position
        curr_pos = np.array([0.0, 0.0, 0.65]) 
        
        print(f"Starting simulation. Trajectory length: {len(trajectory)}")
        for i in range(len(trajectory)):
            delta = trajectory[i][:3] / ACTION_SCALE
            target_pos = curr_pos + delta
            
            # Solve IK
            joint_angles = self.ik_solver.solve_ik(target_pos)
            
            # Map ikpy angles (which includes base/end links) to MuJoCo
            # Our URDF has 4 joints. ikpy chain length might be different.
            # Let's inspect joint_angles size.
            ik_angles = joint_angles[1:5] if len(joint_angles) >= 5 else joint_angles
            
            curr_pos = target_pos
            
            # Step simulation
            for _ in range(5): # 5 frames per waypoint
                # Ensure we don't exceed qpos size
                n_joints = min(len(ik_angles), self.model.nq)
                self.data.qpos[:n_joints] = ik_angles[:n_joints]
                mujoco.mj_step(self.model, self.data)
                
                # Render from a fixed viewpoint since we don't have a camera in the URDF
                # We can use mjv_updateScene with a custom camera
                scn = mujoco.MjvScene(self.model, maxgeom=100)
                cam = mujoco.MjvCamera()
                cam.distance = 1.5
                cam.azimuth = 90
                cam.elevation = -20
                cam.lookat = np.array([0, 0, 0.4])
                
                self.renderer.update_scene(self.data, camera=cam)
                frame = self.renderer.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
        out.release()
        print(f"Simulation saved to {output_video}")

if __name__ == "__main__":
    data_path = "data/processed_bridge.parquet"
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        row = df.iloc[0]
        
        sim = MuJoCoRobotSim(
            "urdf/robot.urdf", 
            "robotmodel/models/checkpoints/latest.pt"
        )
        traj = sim.get_trajectory(row)
        sim.run_simulation(traj, "viz/mujoco_simulation.mp4")
    else:
        print(f"Data path {data_path} not found.")
