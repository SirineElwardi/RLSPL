import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from agent import Agent
from train import train, save
import hyperparameters
import watchDEEP
import WatchTAB

# ---------- Config ----------
ENV_NAME = 'BipedalWalker-v3' 'LunarLander-v2' 'LunarLanderContinuous-v2' 'MountainCar-v0'
SEED = 0
EVAL = True

# ---------- Device ----------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# ---------- Environment ----------
env= gym.make(ENV_NAME, render_mode=None)
env.reset(seed=SEED)
env.action_space.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low

# Extract environment specs
state_dim = env.observation_space.shape[0]

if isinstance(env.action_space, gym.spaces.Box):
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
else:
    action_dim = env.action_space.n
    max_action = None

threshold = env.spec.reward_threshold if env.spec.reward_threshold else None
max_steps = env.spec.max_episode_steps if hasattr(env.spec, "max_episode_steps") else 1000

# ---------- Agent Selection  ----------

# DQN or DDPG (discrete vs continuous)
agent = Agent(state_dim, action_dim)

# DDPG (for continuous environments like LunarLanderContinuous, BipedalWalker)
agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=8)

# SAC (continuous)
agent = Agent(state_dim, env.action_space, device=device)

# Q-learning (MountainCar only)
buckets = (12, 12)
agent = Agent(buckets, hyperparameters.num_episodes, hyperparameters.min_eps, hyperparameters.GAMMA, hyperparameters.EPSILON_DECAY, hyperparameters.LEARNING_RATE, hyperparameters.print_every, env)

# ---------- Training ----------

scores, avg_scores = train(env, agent, threshold, max_steps=max_steps)
scores, avg_scores = train(env, agent, upper_bounds, lower_bounds)



# ---------- Saving ----------
save_dir = f'dir_chk_{ENV_NAME.replace("-", "")}'
save(agent, save_dir, ENV_NAME)

# ---------- Plot ----------
fig, ax = plt.subplots()
ax.plot(np.arange(1, len(scores)+1), scores, label="Score")
ax.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
ax.legend(bbox_to_anchor=(1.05, 1))
ax.set_ylabel('Score')
ax.set_xlabel('Episodes #')
fig.savefig('scores_avg_scores_plot.png', bbox_inches='tight')
plt.close(fig)
print('Graph saved as scores_avg_scores_plot.png')

# ---------- Watch ----------
watchDEEP.visualise()
WatchTAB.visualise(env, agent, 20)

env.close()
print("Training completed.")