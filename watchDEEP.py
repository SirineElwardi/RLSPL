import gymnasium as gym
import torch
from agent import *
import time
import numpy as np
from collections import deque
import hyperparameters
import logging
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(agent, Directory, filename):
    # DQN
    agent.q_local.load_state_dict(torch.load(f'{Directory}/{filename}_local.pth'))
    agent.q_target.load_state_dict(torch.load(f'{Directory}/{filename}_target.pth'))
    # DDPG
    agent.actor_local.load_state_dict(torch.load(f'{Directory}/{filename}_actor_local.pth'))
    agent.actor_target.load_state_dict(torch.load(f'{Directory}/{filename}_actor_target.pth'))
    # SAC
    agent.policy.load_state_dict(torch.load(f'{Directory}/{filename}_actor.pth'))
    agent.critic.load_state_dict(torch.load(f'{Directory}/{filename}_critic.pth'))


def play(env, agent, n_episodes):
    play_log_filename = f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    play_logger = logging.getLogger('play_logger')
    play_logger.setLevel(logging.INFO)
    play_file_handler = logging.FileHandler(play_log_filename)
    play_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    play_logger.addHandler(play_file_handler)

    scores_deque = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        agent.reset()
        total_reward = 0
        time_start = time.time()
        timesteps = 0
        done = False

        while not done:
            env.render()

            # Use only one agent line depending on what you're testing
            action = agent.get_action(torch.FloatTensor([state]), check_eps=False, eps=0.01)  # DQN
            action = agent.get_action(state, add_noise=False)  # DDPG
            action = agent.get_action(state, eval=True)  # SAC

            next_state, reward, terminated, truncated, _ = env.step(action.item() if hasattr(action, 'item') else action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            timesteps += 1

        delta = int(time.time() - time_start)
        scores_deque.append(total_reward)

        play_logger.info(
            f'Episode {i_episode} | Avg Score: {np.mean(scores_deque):.2f} | Timesteps: {timesteps} | Time: {delta//3600:02}:{(delta%3600)//60:02}:{delta%60:02}'
        )


def visualise():
    seed = 0
    env = gym.make('BipedalWalker-v3' 'LunarLander-v2' 'LunarLanderContinuous-v2' 'MountainCar-v0', render_mode="human")  # Replace with target env
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])

    # Instantiate one agent only (match the one you trained)
    agent = Agent(state_dim, action_dim)  # DQN
    agent = Agent(state_dim, action_dim, random_seed=8)  # DDPG
    agent = Agent(state_dim, env.action_space, device=device)  # SAC

    load(agent, 'dir_chk_BW', 'BipedalWalker-v2')  # Change based on what you're loading
    play(env, agent, n_episodes=10)
    env.close()