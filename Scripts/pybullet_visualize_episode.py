import sys
sys.path.append('src')

import time
from stable_baselines3 import SAC
from envs.drone_env import DroneTrackingEnv

def watch_episode(model_path, max_steps=500, speed=1.0):
    env = DroneTrackingEnv(render_mode='rgb_array')
    model = SAC.load(model_path)
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()  # Ensures PyBullet GUI gets updated
        time.sleep(0.05 / speed)  # Adjust speed: lower=faster, higher=slower
        done = terminated or truncated
        step += 1

    env.close()
    print("Visualization complete.")

if __name__ == "__main__":
    watch_episode("models/best_model.zip")
