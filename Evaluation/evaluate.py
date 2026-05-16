import numpy as np
from stable_baselines3 import SAC
from src.envs.drone_env import DroneTrackingEnv
import matplotlib.pyplot as plt

def evaluate_agent(model_path, num_episodes=100):
    """Evaluate trained agent"""
    env = DroneTrackingEnv(render_mode='rgb_array')
    model = SAC.load(model_path)
    
    success_rate = 0
    episode_rewards = []
    tracking_distances = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        distances = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Track distance
            distance = np.linalg.norm(obs['relative_position'])
            distances.append(distance)
            
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        tracking_distances.append(np.mean(distances))
        
        # Success if final distance < 5m
        if distances[-1] < 5.0:
            success_rate += 1
    
    success_rate = (success_rate / num_episodes) * 100
    
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Tracking Distance: {np.mean(tracking_distances):.2f}m")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(tracking_distances)
    plt.title('Average Tracking Distance')
    plt.xlabel('Episode')
    plt.ylabel('Distance (m)')
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    
    return success_rate, episode_rewards, tracking_distances

if __name__ == "__main__":
    evaluate_agent('./models/cuas_agent_final.zip')
