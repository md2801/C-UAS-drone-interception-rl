import sys
sys.path.append('src')

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.drone_env import DroneTrackingEnv
import os

def make_env():
    env = DroneTrackingEnv()
    env = Monitor(env)
    return env

def train_quick():
    """Quick training for 2-day deadline"""
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎮 Training on: {device}")
    
    # Create directories
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create environments
    train_env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel envs
    eval_env = DummyVecEnv([make_env])
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create SAC agent with faster learning settings
    print("🤖 Creating SAC agent...")
    model = SAC(
        'MultiInputPolicy',
        train_env,
        learning_rate=5e-4,  # Higher LR for faster learning
        buffer_size=100000,  # Smaller buffer for faster training
        learning_starts=1000,  # Start learning earlier
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        verbose=1,
        device=device,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Train for shorter time (adjust based on your timeline)
    total_steps = 100000
    print(f"🏋️ Training for {total_steps} steps...")
    
    model.learn(
        total_timesteps=total_steps,
        callback=eval_callback,
        log_interval=10
    )
    
    # Save final model
    model.save("models/cuas_agent_quick")
    print("✅ Training complete! Model saved to models/cuas_agent_quick.zip")
    
    return model

if __name__ == "__main__":
    train_quick()
