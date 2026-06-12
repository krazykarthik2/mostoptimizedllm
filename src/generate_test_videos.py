import os
import mujoco
import imageio
import numpy as np

def generate_video(urdf_path, output_path, task_name, duration_steps=60):
    os.environ['MUJOCO_GL'] = 'egl'
    mj_model = mujoco.MjModel.from_xml_path(urdf_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    
    frames = []
    eef_poses = []
    
    for i in range(duration_steps):
        # Apply some motion based on task
        if "move to target" in task_name:
            mj_data.ctrl[0] = 0.5 * np.sin(i * 0.1)
        elif "lift" in task_name:
            mj_data.ctrl[1] = -0.5 * (i / duration_steps)
        elif "close gripper" in task_name:
            mj_data.ctrl[3] = 1.0 # Assuming 4th is gripper
        else:
            mj_data.ctrl[0] = 0.2 * (i / duration_steps)
            
        mujoco.mj_step(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())
        
        # Track EEF pose [x, y, z, g]
        # ee_link is index 4 (base, l1, l2, l3, ee)
        pos = mj_data.xpos[mj_model.body('ee_link').id]
        gripper = mj_data.ctrl[3] if len(mj_data.ctrl) > 3 else 0
        eef_poses.append(np.append(pos, gripper))
        
    imageio.mimsave(output_path, frames, fps=20)
    print(f"Generated {output_path} for task: {task_name}")
    return np.array(eef_poses)

if __name__ == "__main__":
    os.makedirs("test_videos", exist_ok=True)
    tasks = [
        "move to target",
        "lift object",
        "close gripper",
        "move left",
        "move down"
    ]
    gt_data = {}
    for i, task in enumerate(tasks):
        poses = generate_video("urdf/robot.urdf", f"test_videos/test_{i}.mp4", task)
        gt_data[f"test_{i}"] = {"instruction": task, "poses": poses.tolist()}
    
    import json
    with open("test_videos/gt.json", "w") as f:
        json.dump(gt_data, f)
