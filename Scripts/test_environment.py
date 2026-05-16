"""Quick test to verify environment setup"""
import sys
sys.path.append('src')

from envs.drone_env import DroneTrackingEnv
import numpy as np

def test_environment():
    print("Testing environment creation...")
    env = DroneTrackingEnv()
    
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    print("\nTesting 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print("\n✅ Environment test passed!")
    env.close()

if __name__ == "__main__":
    test_environment()
