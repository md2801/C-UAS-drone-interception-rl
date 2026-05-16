import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data

class DroneTrackingEnv(gym.Env):
    """
    Custom Environment for Drone Tracking using RL
    Based on research from [source:6][source:9]
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Define action space: [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-5, -5, -2, -1]),
            high=np.array([5, 5, 2, 1]),
            dtype=np.float32
        )
        
        # Define observation space: relative position, velocity, image features
        self.observation_space = spaces.Dict({
            'relative_position': spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            'relative_velocity': spaces.Box(low=-20, high=20, shape=(3,), dtype=np.float32),
            'chaser_velocity': spaces.Box(low=-20, high=20, shape=(3,), dtype=np.float32),
            'image_features': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        })
        
        self.render_mode = render_mode
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize PyBullet
        if not hasattr(self, 'physics_client'):
            if self.render_mode == "human":
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
        
        # Reset simulation
        p.resetSimulation()
        self.plane = p.loadURDF("plane.urdf")
        
        # Spawn intruder drone at random position
        self.intruder_pos = np.random.uniform([-50, -50, 10], [50, 50, 30])
        self.intruder_vel = np.random.uniform([-2, -2, -0.5], [2, 2, 0.5])
        
        # Spawn chaser drone
        self.chaser_pos = np.array([0.0, 0.0, 15.0])
        self.chaser_vel = np.array([0.0, 0.0, 0.0])
        
        # --- NEW: Load visible 3D models ---
        self.chaser_id = p.loadURDF("sphere2.urdf", self.chaser_pos.tolist(), globalScaling=3.0)
        self.intruder_id = p.loadURDF("sphere2.urdf", self.intruder_pos.tolist(), globalScaling=2.0)
        
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # Update chaser drone velocity based on action
        self.chaser_vel = action[:3]
        self.chaser_pos += self.chaser_vel * 0.1  # dt = 0.1

        # Update intruder drone (simple motion model)
        self.intruder_pos += self.intruder_vel * 0.1

        # --- Add these lines to move the visible drone bodies! ---
        p.resetBasePositionAndOrientation(self.chaser_id, self.chaser_pos.tolist(), [0,0,0,1])
        p.resetBasePositionAndOrientation(self.intruder_id, self.intruder_pos.tolist(), [0,0,0,1])

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        distance = np.linalg.norm(self.chaser_pos - self.intruder_pos)
        terminated = distance < 2  # Success: within 2m
        truncated = self.current_step >= self.max_steps or distance > 150

        return self._get_obs(), reward, terminated, truncated, {}

    
    def _get_obs(self):
        relative_pos = self.intruder_pos - self.chaser_pos
        relative_vel = self.intruder_vel - self.chaser_vel
        
        # Simulated camera image (placeholder)
        image = np.zeros((84, 84, 3), dtype=np.uint8)
        
        return {
            'relative_position': relative_pos.astype(np.float32),
            'relative_velocity': relative_vel.astype(np.float32),
            'chaser_velocity': self.chaser_vel.astype(np.float32),
            'image_features': image
        }
    
    def _calculate_reward(self):
        distance = np.linalg.norm(self.chaser_pos - self.intruder_pos)
        
        # Reward components (based on SAC approach from [source:9])
        distance_reward = -distance * 0.1
        velocity_alignment = np.dot(self.chaser_vel, self.intruder_vel) * 0.05
        
        # Bonus for getting close
        proximity_bonus = 10.0 if distance < 5.0 else 0.0
        
        return distance_reward + velocity_alignment + proximity_bonus
    
    def get_rgb_frame(self, width=1920, height=1080):
        # Set camera parameters as needed
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0, -80, 30],
            cameraTargetPosition=[0, 0, 15],
            cameraUpVector=[0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width)/height, nearVal=0.1, farVal=200.0)
        (_, _, px, _, _) = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgb_array = np.array(px)[:, :, :3]  # Only RGB, drop alpha
        return rgb_array
    
    def render(self):
        pass
    
    def close(self):
        if hasattr(self, 'physics_client'):
            p.disconnect()
