import random
from collections import namedtuple , deque
import torch
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
device = torch.device("cpu")




class ReplayMemory(object):

#DQN
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

#DQN
    def add(self, batch):
        self.memory.append(batch)
        if len(self.memory) > self.capacity:
            del self.memory[0]    
     

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

#DDPG
    def __init__(self, action_size, buffer_size, batch_size, seed):
       
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

#DDPG
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

#SAC
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        

    def add(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
    
        return state, action, reward, next_state, done


#common 
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)