import gym
from gym import spaces
import numpy as np

class CustomENV(gym.Env):
    def __init__(self):
       
        # Define your custom observation and action spaces if needed
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # Example discrete action space

    def step(self, action):
        # Override the default step method to customize the reward calculation
        observation, reward, done, info = self.env.step(action)
        # Custom reward calculation
        custom_reward = self.custom_reward(observation, reward, done, info)
        return observation, custom_reward, done, info

    def custom_reward(self, observation, reward, done, info):
        # Implement your custom reward logic here
        custom_reward = reward  # Example: Custom reward is the same as the default reward
        return custom_reward

    def reset(self):
        # Override reset method if needed
        return self.env.reset()

    def render(self, mode='human'):
        # Override render method if needed
        return self.env.render(mode)

    def close(self):
        # Override close method if needed
        return self.env.close()


class CustomENV(BaseEnv):
    def __init__(self, num_servers=5, job_queue_size=10):
        self.num_servers = num_servers
        self.job_queue = np.random.randint(1, 10, size=job_queue_size)  # Random job sizes
        self.state = self._initialize_state()

    def reset(self):
        self.state = self._initialize_state()
        return self.state

    def step(self, action):
        reward, done = self._apply_action(action)
        next_state = self.get_state()
        return next_state, reward, done

    def get_state(self):
        return self.state

    def get_action_space(self):
        return list(range(self.num_servers))  # Available servers

    def _initialize_state(self):
        return {"job_queue": self.job_queue.copy(), "server_load": np.zeros(self.num_servers)}

    def _apply_action(self, action):
        """Distribute a job to a selected server and compute reward."""
        if len(self.job_queue) == 0:
            return -1, True  # No jobs left, terminate
        job_size = self.job_queue[0]
        self.job_queue = self.job_queue[1:]  # Remove processed job
        self.state["server_load"][action] += job_size  # Assign job to selected server
        reward = -self.state["server_load"][action]  # Negative reward for high load
        return reward, False