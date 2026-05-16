"""
DQN (Deep Q-Network) training script for Drone Tracking
"""
import os
import sys

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.envs.drone_env_discrete import DroneTrackingEnvDiscrete


class DQNTrainer:
    """
    Trainer for CUAS RL Agent using DQN (Deep Q-Network)
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")
        
    def create_env(self):
        """Create vectorized environment"""
        def make_env():
            env = DroneTrackingEnvDiscrete()
            env = Monitor(env)
            return env
        
        # DQN typically works better with single environment or small number of parallel envs
        if self.config.get('num_envs', 1) > 1:
            env = SubprocVecEnv([make_env for _ in range(self.config['num_envs'])])
        else:
            env = DummyVecEnv([make_env])
        
        return env
    
    def train(self):
        """Main training loop"""
        # Create environments
        train_env = self.create_env()
        eval_env = self.create_env()
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./models/checkpoints/',
            name_prefix='dqn_cuas_agent'
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/best_dqn/',
            log_path='./logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Initialize DQN agent
        # Note: DQN uses MultiInputPolicy for Dict observation spaces
        model = DQN(
            'MultiInputPolicy',
            train_env,
            learning_rate=1e-4,  # Lower learning rate for DQN
            buffer_size=50000,  # Replay buffer size
            learning_starts=5000,  # Start learning after collecting some samples
            batch_size=32,  # Batch size for DQN (typically smaller than SAC)
            tau=1.0,  # Hard target update (DQN uses hard updates by default)
            gamma=0.99,  # Discount factor
            train_freq=(4, "step"),  # Train every 4 steps
            gradient_steps=1,  # Number of gradient steps per update
            target_update_interval=1000,  # Update target network every 1000 steps
            exploration_fraction=0.1,  # Fraction of timesteps for exploration
            exploration_initial_eps=1.0,  # Initial epsilon
            exploration_final_eps=0.05,  # Final epsilon
            max_grad_norm=10,  # Gradient clipping
            tensorboard_log="./tensorboard/",
            device=self.device,
            verbose=1
        )
        
        # Train the model
        print(f"Starting DQN training for {self.config['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            tb_log_name="DQN"
        )
        
        # Save final model
        model.save("./models/dqn_cuas_agent_final")
        print("DQN training completed!")
        
        return model


if __name__ == "__main__":
    config = {
        'num_envs': 1,  # DQN typically uses single environment
        'total_timesteps': 500000,  # Can adjust based on needs
    }
    
    trainer = DQNTrainer(config)
    model = trainer.train()

