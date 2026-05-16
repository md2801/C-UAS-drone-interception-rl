"""
Quick test to verify DQN discrete environment setup
"""
import sys
import os

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.envs.drone_env_discrete import DroneTrackingEnvDiscrete
import numpy as np

def test_discrete_environment():
    print("Testing discrete environment creation...")
    env = DroneTrackingEnvDiscrete()
    
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    print(f"Observation space: {env.observation_space}")
    
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    
    print("\nTesting discrete action mapping...")
    # Test a few discrete actions
    test_actions = [0, 13, 26]  # First, middle, last action
    for action in test_actions:
        continuous = env._discrete_to_continuous(action)
        print(f"Action {action} -> Continuous: {continuous}")
    
    print("\nTesting 10 random discrete steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            obs, info = env.reset()
            print("  Episode ended, resetting...")
    
    print("\n[OK] Discrete environment test passed!")
    env.close()

if __name__ == "__main__":
    test_discrete_environment()

