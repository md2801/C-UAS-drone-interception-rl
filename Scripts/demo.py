"""Demo script to visualize trained agent"""
import sys
import os
sys.path.append('src')

import numpy as np
from stable_baselines3 import SAC
from envs.drone_env import DroneTrackingEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def demo_agent(model_path, render=True):
    """Run visual demo of trained agent"""
    env = DroneTrackingEnv(render_mode='rgb_array' if render else None)
    model = SAC.load(model_path)
    
    obs, _ = env.reset()
    
    # Track trajectories
    chaser_trajectory = [env.chaser_pos.copy()]
    intruder_trajectory = [env.intruder_pos.copy()]
    
    done = False
    step = 0
    
    print("🎬 Running demo...")
    while not done and step < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        chaser_trajectory.append(env.chaser_pos.copy())
        intruder_trajectory.append(env.intruder_pos.copy())
        
        done = terminated or truncated
        step += 1
        
        if step % 50 == 0:
            distance = np.linalg.norm(obs['relative_position'])
            print(f"Step {step}: Distance = {distance:.2f}m")
    
    env.close()
    
    # Visualize trajectories
    chaser_trajectory = np.array(chaser_trajectory)
    intruder_trajectory = np.array(intruder_trajectory)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(intruder_trajectory[:, 0], intruder_trajectory[:, 1], 
             intruder_trajectory[:, 2], 'r-', label='Intruder', linewidth=2)
    ax1.plot(chaser_trajectory[:, 0], chaser_trajectory[:, 1], 
             chaser_trajectory[:, 2], 'b-', label='Chaser (RL)', linewidth=2)
    ax1.scatter(intruder_trajectory[0, 0], intruder_trajectory[0, 1], 
                intruder_trajectory[0, 2], c='red', s=100, marker='o')
    ax1.scatter(chaser_trajectory[0, 0], chaser_trajectory[0, 1], 
                chaser_trajectory[0, 2], c='blue', s=100, marker='o')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # Distance over time
    ax2 = fig.add_subplot(122)
    distances = np.linalg.norm(chaser_trajectory - intruder_trajectory, axis=1)
    ax2.plot(distances, linewidth=2)
    ax2.axhline(y=5.0, color='g', linestyle='--', label='Success threshold')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Tracking Distance Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/demo_trajectory.png', dpi=150)
    print("📊 Demo visualization saved to results/demo_trajectory.png")
    plt.show()
    
    final_distance = distances[-1]
    success = final_distance < 5.0
    print(f"\n{'='*50}")
    print(f"Demo Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Final Distance: {final_distance:.2f}m")
    print(f"Steps taken: {step}")
    print(f"{'='*50}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/best_model.zip"
    demo_agent(model_path)
