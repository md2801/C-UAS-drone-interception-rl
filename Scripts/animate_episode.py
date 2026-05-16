import sys
sys.path.append('src')

import numpy as np
from stable_baselines3 import SAC
from envs.drone_env import DroneTrackingEnv
import matplotlib.pyplot as plt
from matplotlib import animation

def run_episode_and_record(model_path, max_steps=150):
    env = DroneTrackingEnv(render_mode=None)
    model = SAC.load(model_path)

    obs, _ = env.reset()
    chaser_positions = [env.chaser_pos.copy()]
    intruder_positions = [env.intruder_pos.copy()]
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        chaser_positions.append(env.chaser_pos.copy())
        intruder_positions.append(env.intruder_pos.copy())
        done = terminated or truncated
        step += 1

    env.close()
    return np.array(intruder_positions), np.array(chaser_positions)

def animate_trajectory(intruder, chaser, save_path='results/cuad_demo.mp4'):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min(np.min(intruder[:,0]), np.min(chaser[:,0]))-5, max(np.max(intruder[:,0]), np.max(chaser[:,0]))+5)
    ax.set_ylim(min(np.min(intruder[:,1]), np.min(chaser[:,1]))-5, max(np.max(intruder[:,1]), np.max(chaser[:,1]))+5)
    ax.set_zlim(min(np.min(intruder[:,2]), np.min(chaser[:,2]))-5, max(np.max(intruder[:,2]), np.max(chaser[:,2]))+5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    intruder_line, = ax.plot([], [], [], 'r-', label='Intruder')
    chaser_line,   = ax.plot([], [], [], 'b-', label='Chaser')
    intruder_dot, = ax.plot([], [], [], 'ro', markersize=8)
    chaser_dot, = ax.plot([], [], [], 'bo', markersize=8)
    ax.legend()

    def init():
        intruder_line.set_data([], [])
        intruder_line.set_3d_properties([])
        chaser_line.set_data([], [])
        chaser_line.set_3d_properties([])
        return intruder_line, chaser_line, intruder_dot, chaser_dot

    def update(num):
        intruder_line.set_data(intruder[:num+1, 0], intruder[:num+1, 1])
        intruder_line.set_3d_properties(intruder[:num+1, 2])
        chaser_line.set_data(chaser[:num+1, 0], chaser[:num+1, 1])
        chaser_line.set_3d_properties(chaser[:num+1, 2])

        intruder_dot.set_data([intruder[num, 0]], [intruder[num, 1]])
        intruder_dot.set_3d_properties([intruder[num, 2]])
        chaser_dot.set_data([chaser[num, 0]], [chaser[num, 1]])
        chaser_dot.set_3d_properties([chaser[num, 2]])
        return intruder_line, chaser_line, intruder_dot, chaser_dot

    ani = animation.FuncAnimation(
        fig, update, frames=len(intruder), init_func=init, blit=True, interval=60, repeat=False
    )

    ani.save(save_path, writer='ffmpeg', fps=24)
    print(f"Video saved to {save_path}")

if __name__ == "__main__":
    # Edit below: provide the correct path to your trained model
    intruder, chaser = run_episode_and_record('models/best_model.zip')
    animate_trajectory(intruder, chaser, save_path='results/cuad_demo.mp4')
