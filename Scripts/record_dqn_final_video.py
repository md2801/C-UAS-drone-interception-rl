"""
Final video recording for DQN trained agent
Records high-quality video from PyBullet for final submission
"""
import sys
import os
import numpy as np
import cv2

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stable_baselines3 import DQN
from src.envs.drone_env_discrete import DroneTrackingEnvDiscrete

def record_dqn_final_video(model_path, out_path="results/dqn_final_episode.mp4", 
                          max_steps=1000, fps=30, width=1920, height=1080):
    """
    Record final submission video of DQN agent
    
    Args:
        model_path: Path to trained DQN model
        out_path: Output video path
        max_steps: Maximum steps per episode
        fps: Video frame rate
        width: Video width
        height: Video height
    """
    print(f"Loading DQN model from {model_path}...")
    model = DQN.load(model_path)
    
    # Use 'human' mode to enable PyBullet GUI rendering
    env = DroneTrackingEnvDiscrete(render_mode="human")
    obs, info = env.reset()
    
    frames = []
    rewards = []
    distances = []
    
    print("Recording episode...")
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture frame
        frame = env.get_rgb_frame(width=width, height=height)
        frames.append(frame)
        
        # Track metrics
        distance = np.linalg.norm(obs['relative_position'])
        rewards.append(reward)
        distances.append(distance)
        
        if step % 100 == 0:
            print(f"  Step {step}/{max_steps}, Distance: {distance:.2f}m, Reward: {reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}: {'SUCCESS' if terminated else 'TRUNCATED'}")
            break
    
    env.close()
    
    # Save video
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    print(f"Writing {len(frames)} frames to video...")
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    
    video.release()
    
    print(f"\n{'='*60}")
    print("FINAL VIDEO RECORDING COMPLETE")
    print(f"{'='*60}")
    print(f"Video saved to: {out_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Duration: {len(frames)/fps:.2f} seconds")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Final distance: {distances[-1]:.2f}m")
    print(f"Minimum distance: {min(distances):.2f}m")
    print(f"{'='*60}")

if __name__ == "__main__":
    model_path = "models/best_dqn/best_model.zip"
    record_dqn_final_video(model_path, out_path="results/dqn_final_submission.mp4")