"""
Demo script for DQN trained agent
Loads a DQN model and visualizes its performance
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stable_baselines3 import DQN
from src.envs.drone_env_discrete import DroneTrackingEnvDiscrete


def demo_dqn_agent(model_path, num_episodes=5, render=True):
    """
    Run episodes with trained DQN agent and visualize trajectories
    
    Args:
        model_path: Path to trained DQN model
        num_episodes: Number of episodes to run
        render: Whether to show plots
    """
    print(f"Loading DQN model from {model_path}...")
    try:
        model = DQN.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    env = DroneTrackingEnvDiscrete()
    
    episode_rewards = []
    episode_lengths = []
    all_trajectories = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Store trajectory
        chaser_positions = [env.chaser_pos.copy()]
        intruder_positions = [env.intruder_pos.copy()]
        
        while not done:
            # Get action from DQN model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Store positions
            chaser_positions.append(env.chaser_pos.copy())
            intruder_positions.append(env.intruder_pos.copy())
            
            if done:
                distance = np.linalg.norm(env.chaser_pos - env.intruder_pos)
                print(f"  Episode ended: reward={total_reward:.2f}, steps={steps}, "
                      f"final_distance={distance:.2f}m, terminated={terminated}")
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        all_trajectories.append((np.array(chaser_positions), np.array(intruder_positions)))
    
    env.close()
    
    # Print statistics
    print(f"\n{'='*50}")
    print("DQN Agent Performance Summary:")
    print(f"{'='*50}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Success rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) * 100:.1f}%")
    
    # Visualize trajectories
    if render and all_trajectories:
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 3D trajectory for first episode
        ax1 = fig.add_subplot(131, projection='3d')
        chaser, intruder = all_trajectories[0]
        ax1.plot(chaser[:, 0], chaser[:, 1], chaser[:, 2], 'b-', label='Chaser', linewidth=2)
        ax1.plot(intruder[:, 0], intruder[:, 1], intruder[:, 2], 'r-', label='Intruder', linewidth=2)
        ax1.scatter(chaser[0, 0], chaser[0, 1], chaser[0, 2], c='blue', s=100, marker='o', label='Start')
        ax1.scatter(chaser[-1, 0], chaser[-1, 1], chaser[-1, 2], c='green', s=100, marker='*', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Episode 1: 3D Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # Plot distance over time for first episode
        ax2 = fig.add_subplot(132)
        distances = [np.linalg.norm(chaser[i] - intruder[i]) for i in range(len(chaser))]
        ax2.plot(distances, linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Distance (m)')
        ax2.set_title('Distance Over Time')
        ax2.grid(True)
        
        # Plot reward distribution
        ax3 = fig.add_subplot(133)
        ax3.bar(range(len(episode_rewards)), episode_rewards, color='green', alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Total Reward')
        ax3.set_title('Episode Rewards')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/dqn_demo_results.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to results/dqn_demo_results.png")
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Demo DQN trained agent')
    parser.add_argument('--model', type=str, default='models/best_dqn/best_model.zip',
                       help='Path to DQN model')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    demo_dqn_agent(args.model, args.episodes, render=not args.no_render)


