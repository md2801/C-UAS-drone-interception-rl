"""
Final comprehensive evaluation for DQN trained agent
"""
import os
import sys

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from stable_baselines3 import DQN
from src.envs.drone_env_discrete import DroneTrackingEnvDiscrete
import matplotlib.pyplot as plt



def evaluate_dqn_final(model_path, num_episodes=100, save_dir='results'):
    """
    Comprehensive final evaluation of DQN agent
    
    Args:
        model_path: Path to trained DQN model
        num_episodes: Number of evaluation episodes
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading DQN model from {model_path}...")
    model = DQN.load(model_path)
    env = DroneTrackingEnvDiscrete(render_mode='rgb_array')
    
    # Metrics to track
    episode_rewards = []
    episode_lengths = []
    final_distances = []
    success_count = 0
    terminated_count = 0  # Success (distance < 0.5m)
    truncated_count = 0   # Failure (timeout or distance > 150m)
    min_distances = []
    all_distance_curves = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        distances = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            distance = np.linalg.norm(obs['relative_position'])
            distances.append(distance)
            
            done = terminated or truncated
        
        # Calculate metrics
        final_distance = distances[-1]
        min_distance = min(distances)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        final_distances.append(final_distance)
        min_distances.append(min_distance)
        all_distance_curves.append(distances)
        
        # Success criteria: final distance < 0.5m (terminated) or < 5m
        if terminated:  # Success (distance < 0.5m)
            success_count += 1
            terminated_count += 1
        elif final_distance < 5.0:  # Close enough
            success_count += 1
        else:
            truncated_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")
    
    env.close()
    
    # Calculate statistics
    success_rate = (success_count / num_episodes) * 100
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_final_distance = np.mean(final_distances)
    avg_min_distance = np.mean(min_distances)
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print("FINAL DQN EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_episodes})")
    print(f"  - Successful Interceptions (terminated): {terminated_count}")
    print(f"  - Failed Episodes (truncated): {truncated_count}")
    print(f"\nEpisode Statistics:")
    print(f"  Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Average Episode Length: {avg_length:.1f} steps")
    print(f"  Average Final Distance: {avg_final_distance:.2f}m")
    print(f"  Average Minimum Distance: {avg_min_distance:.2f}m")
    print(f"  Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"  Worst Episode Reward: {min(episode_rewards):.2f}")
    print(f"{'='*60}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, alpha=0.7, linewidth=1)
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Mean: {avg_reward:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Final Distances
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(final_distances, alpha=0.7, linewidth=1)
    plt.axhline(y=0.5, color='g', linestyle='--', label='Success (0.5m)')
    plt.axhline(y=5.0, color='orange', linestyle='--', label='Good (5m)')
    plt.axhline(y=avg_final_distance, color='r', linestyle='--', label=f'Mean: {avg_final_distance:.2f}m')
    plt.xlabel('Episode')
    plt.ylabel('Final Distance (m)')
    plt.title('Final Tracking Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Episode Lengths
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(episode_lengths, alpha=0.7, linewidth=1)
    plt.axhline(y=avg_length, color='r', linestyle='--', label=f'Mean: {avg_length:.1f}')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Distance curves (first 10 episodes)
    ax4 = plt.subplot(2, 3, 4)
    for i in range(min(10, len(all_distance_curves))):
        plt.plot(all_distance_curves[i], alpha=0.5, linewidth=1)
    plt.axhline(y=0.5, color='g', linestyle='--', label='Success (0.5m)')
    plt.xlabel('Step')
    plt.ylabel('Distance (m)')
    plt.title('Distance Over Time (First 10 Episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Reward Distribution Histogram
    ax5 = plt.subplot(2, 3, 5)
    plt.hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=avg_reward, color='r', linestyle='--', label=f'Mean: {avg_reward:.2f}')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Success/Failure Pie Chart
    ax6 = plt.subplot(2, 3, 6)
    success_fail = [success_count, num_episodes - success_count]
    labels = [f'Success\n({success_count})', f'Failure\n({num_episodes - success_count})']
    colors = ['green', 'red']
    plt.pie(success_fail, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Success Rate: {success_rate:.1f}%')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dqn_final_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_dir}/dqn_final_evaluation.png")
    
    # Save metrics to text file
    with open(f'{save_dir}/dqn_final_metrics.txt', 'w') as f:
        f.write("DQN Final Evaluation Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluation Episodes: {num_episodes}\n\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n")
        f.write(f"Successful Interceptions: {terminated_count}\n")
        f.write(f"Failed Episodes: {truncated_count}\n\n")
        f.write(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}\n")
        f.write(f"Average Episode Length: {avg_length:.1f} steps\n")
        f.write(f"Average Final Distance: {avg_final_distance:.2f}m\n")
        f.write(f"Average Minimum Distance: {avg_min_distance:.2f}m\n")
        f.write(f"Best Episode Reward: {max(episode_rewards):.2f}\n")
        f.write(f"Worst Episode Reward: {min(episode_rewards):.2f}\n")
    
    print(f"Metrics saved to {save_dir}/dqn_final_metrics.txt")
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'avg_final_distance': avg_final_distance,
        'avg_min_distance': avg_min_distance
    }

if __name__ == "__main__":
    model_path = "models/best_dqn/best_model.zip"
    evaluate_dqn_final(model_path, num_episodes=100)