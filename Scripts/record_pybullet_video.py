import sys, os
sys.path.append('src')

import numpy as np
import cv2
from stable_baselines3 import SAC
from envs.drone_env import DroneTrackingEnv

def record_episode(model_path, out_path="results/pybullet_episode.mp4", max_steps=500, fps=24):
    env = DroneTrackingEnv(render_mode="rgb_array")
    model = SAC.load(model_path)
    obs, _ = env.reset()
    frames = []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.get_rgb_frame()  # Use the function from Step 1
        frames.append(frame)
        if terminated or truncated:
            break
    env.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for img in frames:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    record_episode("models/best_model.zip")
