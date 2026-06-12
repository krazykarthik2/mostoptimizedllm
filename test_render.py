import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import mujoco
import numpy as np

try:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='plane' size='1 1 .01'/></worldbody></mujoco>")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    renderer.update_scene(data)
    frame = renderer.render()
    print(f"Successfully rendered frame of shape {frame.shape}")
except Exception as e:
    print(f"Rendering failed: {e}")
