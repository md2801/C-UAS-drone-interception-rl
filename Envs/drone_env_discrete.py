"""
Discrete action space wrapper for DroneTrackingEnv
Converts discrete actions to continuous velocity commands for DQN compatibility
"""
import numpy as np
from gymnasium import spaces
from .drone_env import DroneTrackingEnv


class DroneTrackingEnvDiscrete(DroneTrackingEnv):
    """
    Discrete action space version of DroneTrackingEnv for DQN
    
    Action space: 27 discrete actions mapping to velocity combinations
    Actions represent combinations of:
    - vx: [-5, 0, 5] m/s (3 levels)
    - vy: [-5, 0, 5] m/s (3 levels)  
    - vz: [-2, 0, 2] m/s (3 levels)
    - Total: 3^3 = 27 actions
    """
    
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # Define discrete action space (27 actions)
        self.action_space = spaces.Discrete(27)
        
        # Action mapping: 27 discrete actions to [vx, vy, vz, yaw_rate]
        # Structure: action = vx_idx * 9 + vy_idx * 3 + vz_idx
        # where each idx is 0, 1, 2 mapping to [-val, 0, val]
        self.vx_levels = np.array([-5.0, 0.0, 5.0])
        self.vy_levels = np.array([-5.0, 0.0, 5.0])
        self.vz_levels = np.array([-2.0, 0.0, 2.0])
        self.yaw_rate = 0.0  # Fixed yaw rate for discrete version
        
    def _discrete_to_continuous(self, action):
        """
        Convert discrete action (0-26) to continuous action [vx, vy, vz, yaw_rate]
        
        Args:
            action: Integer in range [0, 26]
            
        Returns:
            np.array: [vx, vy, vz, yaw_rate]
        """
        # Decompose action into indices
        vx_idx = action // 9
        vy_idx = (action // 3) % 3
        vz_idx = action % 3
        
        # Map indices to velocity values
        vx = self.vx_levels[vx_idx]
        vy = self.vy_levels[vy_idx]
        vz = self.vz_levels[vz_idx]
        
        return np.array([vx, vy, vz, self.yaw_rate], dtype=np.float32)
    
    def step(self, action):
        """
        Execute discrete action by converting to continuous and calling parent step
        
        Args:
            action: Integer discrete action (0-26)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert discrete action to continuous
        continuous_action = self._discrete_to_continuous(action)
        
        # Call parent step with continuous action
        return super().step(continuous_action)
    
    def get_action_meanings(self):
        """
        Get human-readable meaning of each discrete action
        
        Returns:
            list: List of action descriptions
        """
        meanings = []
        for action in range(27):
            vx_idx = action // 9
            vy_idx = (action // 3) % 3
            vz_idx = action % 3
            
            vx = self.vx_levels[vx_idx]
            vy = self.vy_levels[vy_idx]
            vz = self.vz_levels[vz_idx]
            
            meanings.append(f"vx={vx:.1f}, vy={vy:.1f}, vz={vz:.1f}")
        
        return meanings


