import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TelescopeSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_telescopes = 3
        self.num_observations = 5

        # Observation space is a dictionary of telescope and observation features
        self.observation_space = spaces.Dict({
            "telescope_status": spaces.MultiBinary(self.num_telescopes),
            "observation_priority": spaces.Box(low=0, high=1, shape=(self.num_observations,), dtype=np.float32),
            "cloud_cover": spaces.Box(low=0, high=1, shape=(self.num_telescopes,), dtype=np.float32),
        })

        # Action space is selecting telescope-observation pairs or a no-op
        self.action_space = spaces.Discrete(self.num_telescopes * self.num_observations + 1)  # last is "do nothing"

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.telescope_status = [1] * self.num_telescopes
        self.observation_priority = np.random.choice([0.2, 0.5, 0.8, 1.0], size=self.num_observations)
        self.cloud_cover = np.random.uniform(0, 1, size=self.num_telescopes)
        self.completed = [0] * self.num_observations
        return self._get_obs(), {}  # <- must return (obs, info)

    def _get_obs(self):
        return {
            "telescope_status": np.array(self.telescope_status, dtype=np.int8),
            "observation_priority": np.array(self.observation_priority, dtype=np.float32),
            "cloud_cover": np.array(self.cloud_cover, dtype=np.float32),
        }

    def step(self, action):
        reward = 0
        done = False

        if action == self.num_telescopes * self.num_observations:
            # No-op
            reward = -0.05  # slight penalty for doing nothing
        else:
            tel_idx = action % self.num_telescopes
            obs_idx = action // self.num_telescopes

            if self.telescope_status[tel_idx] == 1 and self.completed[obs_idx] == 0:
                if self.cloud_cover[tel_idx] < 0.7:
                    reward = self.observation_priority[obs_idx]  # success
                else:
                    reward = -0.5  # failed due to weather
                self.completed[obs_idx] = 1
                self.telescope_status[tel_idx] = 0  # mark telescope busy
            else:
                reward = -0.2  # invalid or redundant action

        # Episode ends if all observations completed or after fixed steps
        done = all(self.completed)
        return self._get_obs(), reward, done, False, {}  # <- 5 elements!

    def render(self, mode="human"):
        print("Telescope Status:", self.telescope_status)
        print("Observation Priorities:", self.observation_priority)
        print("Completed:", self.completed)
        print("Cloud Cover:", self.cloud_cover)
