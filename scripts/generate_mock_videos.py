import os
import imageio
import numpy as np

def generate_mock_video(output_path, task_name, duration=30):
    frames = []
    for i in range(duration):
        # Create a simple image with a moving block
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Move a red square based on time
        y = int(50 + 100 * np.sin(i * 0.2))
        x = int(50 + 100 * np.cos(i * 0.2))
        img[y:y+30, x:x+30, 0] = 255
        frames.append(img)
    imageio.mimsave(output_path, frames, fps=10)
    print(f"Mock video saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("test_videos", exist_ok=True)
    tasks = ["move circle", "move fast", "stationary", "zigzag", "diagonal"]
    for i, t in enumerate(tasks):
        generate_mock_video(f"test_videos/mock_{i}.mp4", t)
