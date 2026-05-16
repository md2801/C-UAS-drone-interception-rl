import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import sys

sys.path.append('src')
from envs.drone_env import DroneTrackingEnv

class CUASTrainer:
    """
    Trainer for CUAS RL Agent
    Based on SAC algorithm (Soft Actor-Critic) from [source:9]
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")
        
    def create_env(self):
        """Create vectorized environment"""
        def make_env():
            env = DroneTrackingEnv()
            env = Monitor(env)
            return env
        
        # Create multiple parallel environments
        if self.config['num_envs'] > 1:
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
            name_prefix='cuas_agent'
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/',
            log_path='./logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Initialize SAC agent
        model = SAC(
            'MultiInputPolicy',
            train_env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto',
            tensorboard_log="./tensorboard/",
            device=self.device,
            verbose=1
        )
        
        # Train the model
        print(f"Starting training for {self.config['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            tb_log_name="SAC"
        )
        
        # Save final model
        model.save("./models/cuas_agent_final")
        print("Training completed!")
        
        return model

if __name__ == "__main__":
    config = {
        'num_envs': 4,
        'total_timesteps': 5000000,
    }
    
    trainer = CUASTrainer(config)
    model = trainer.train()
