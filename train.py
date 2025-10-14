import numpy as np
import torch
import time
from collections import deque
from agent import FloatTensor
from torch.autograd import Variable
import hyperparameters
import logging
from datetime import datetime

torch.autograd.set_detect_anomaly(True)

# ------------------------------
# Epsilon Annealing (DQN)
# ------------------------------
def epsilon_annealing(i_episode, min_eps, max_eps_episode):
    slope = (min_eps - 1.0) / max_eps_episode
    return max(slope * i_episode + 1.0, min_eps)

# ------------------------------
# DQN Training
# ------------------------------
def train(env, agent, threshold):
    train_logger = create_logger()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []    
    time_start = time.time()

    for i_episode in range(1, hyperparameters.num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        eps = epsilon_annealing(i_episode, hyperparameters.min_eps, hyperparameters.max_eps_episode)

        while not done:
            action = agent.get_action(FloatTensor([state]), eps)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            if done:
                reward = -1

            agent.replay_memory.add(
                (FloatTensor([state]), 
                 action,
                 FloatTensor([reward]), 
                 FloatTensor([next_state]), 
                 FloatTensor([done]))
            )

            if len(agent.replay_memory) > hyperparameters.BATCH_SIZE:
                batch = agent.replay_memory.sample(hyperparameters.BATCH_SIZE)
                agent.learn(batch)

            state = next_state

        scores_deque.append(total_reward)
        scores_array.append(total_reward)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        dt = int(time.time() - time_start)

        if i_episode % hyperparameters.print_every == 0 or (len(scores_deque) == 100 and avg_score >= threshold):
            train_logger.info(
                f'Episode {i_episode} Score: {total_reward:.2f} Average Score: {avg_score:.2f}, Time: {dt // 3600:02}:{(dt % 3600) // 60:02}:{dt % 60:02} ***'
            )

        if len(scores_deque) == 100 and avg_score >= threshold:
            train_logger.info('Environment solved!')
            break

        if i_episode % hyperparameters.TARGET_UPDATE == 0:
            agent.q_target.load_state_dict(agent.q_local.state_dict())

    return scores_array, avg_scores_array

# ------------------------------
# DDPG Training
# ------------------------------
def train(env, agent, threshold):
    train_logger = create_logger()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []    
    time_start = time.time()

    for i_episode in range(1, hyperparameters.num_episodes + 1):
        state, _ = env.reset()
        agent.reset()
        total_reward = 0
        timestep = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_bool = 0 if timestep + 1 == env.spec.max_episode_steps else float(done)
            total_reward += reward

            agent.step(state, action, reward, next_state, done, timestep)
            state = next_state
            timestep += 1

        scores_deque.append(total_reward)
        scores_array.append(total_reward)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        dt = int(time.time() - time_start)

        if i_episode % hyperparameters.print_every == 0 or (len(scores_deque) == 100 and avg_score >= threshold):
            train_logger.info(
                f'Episode {i_episode} Score: {total_reward:.2f} Average Score: {avg_score:.2f}, Time: {dt // 3600:02}:{(dt % 3600) // 60:02}:{dt % 60:02} ***'
            )

        if len(scores_deque) == 100 and avg_score >= threshold:
            train_logger.info('Environment solved!')
            break

    return scores_array, avg_scores_array

# ------------------------------
# Q-Learning Training
# ------------------------------
def train(env, agent, upper_bounds, lower_bounds):
    train_logger = create_logger()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []  
    time_start = time.time()

    for i_episode in range(agent.num_episodes):
        obs, _ = env.reset()
        current_state = agent.discretize_state(obs, upper_bounds, lower_bounds)
        epsilon = agent.get_epsilon(i_episode)
        done = False
        episode_reward = 0
        time_step = 0

        while not done:
            action = agent.get_action(current_state, env, epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_state = agent.discretize_state(obs, upper_bounds, lower_bounds)
            agent.update_q(agent.Q_table, current_state, action, reward, new_state, agent.learning_rate, agent.discount)
            current_state = new_state
            time_step += 1
            episode_reward += reward

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        s = int(time.time() - time_start)

        if i_episode % agent.print_every == 0 and i_episode > 0:
            train_logger.info(
                f'Episode: {i_episode}, Timesteps: {time_step}, Score: {episode_reward}, Avg.Score: {avg_score:.2f}, eps-greedy: {epsilon:.2f}, Time: {s//3600:02}:{(s%3600)//60:02}:{s%60:02}'
            )

        if avg_score >= -110:
            train_logger.info(f'\nEnvironment solved in {i_episode} episodes! Average Score: {avg_score:.2f}')
            break

    train_logger.info('Finished training!')
    return scores_array, avg_scores_array

# ------------------------------
# SAC Training
# ------------------------------
def train(env, agent, max_steps):
    train_logger = create_logger()
    total_numsteps = 0
    updates = 0
    start_steps = 10000
    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = [] 

    for i_episode in range(hyperparameters.num_episodes): 
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset()

        for step in range(max_steps):    
            if total_numsteps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)

            if len(agent.replay_memory) > hyperparameters.BATCH_SIZE:
                agent.learn(agent.replay_memory, hyperparameters.BATCH_SIZE, updates)
                updates += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == env.spec.max_episode_steps else float(not done)

            agent.replay_memory.add(state, action, reward, next_state, mask)
            state = next_state

            if done:
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        s = int(time.time() - time_start)

        if i_episode % 20 == 0 and i_episode > 0:
            save(agent, 'dir_chk_BW', 'BipedalWalker-v2')
            train_logger.info(f"Weights saved for episode {i_episode}")

        train_logger.info(
            f"Ep.: {i_episode}, Total Steps: {total_numsteps}, Ep.Steps: {episode_steps}, Score: {episode_reward:.2f}, Avg.Score: {avg_score:.2f}, Time: {s//3600:02}:{(s%3600)//60:02}:{s%60:02}"
        )

        if avg_score > 300.5:
            train_logger.info(f'Solved environment with Avg Score: {avg_score}')
            break

    return scores_array, avg_scores_array

# ------------------------------
# Save Function
# ------------------------------
def save(agent, directory, filename):
    torch.save(agent.actor_local.state_dict(), f'{directory}/{filename}_actor_local.pth')
    torch.save(agent.actor_target.state_dict(), f'{directory}/{filename}_actor_target.pth')
    torch.save(agent.critic_local.state_dict(), f'{directory}/{filename}_critic_local.pth')
    torch.save(agent.critic_target.state_dict(), f'{directory}/{filename}_critic_target.pth')
    torch.save(agent.q_local.state_dict(), f'{directory}/{filename}_local.pth')
    torch.save(agent.q_target.state_dict(), f'{directory}/{filename}_target.pth')
    torch.save(agent.policy.state_dict(), f'{directory}/{filename}_actor.pth')
    torch.save(agent.critic.state_dict(), f'{directory}/{filename}_critic.pth')

# ------------------------------
# Logger Setup
# ------------------------------
def create_logger():
    log_name = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = logging.getLogger(f'training_logger_{log_name}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_name)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger