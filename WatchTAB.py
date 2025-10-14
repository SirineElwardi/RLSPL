import logging
from datetime import datetime
import numpy as np

def visualise(env, agent, n_episodes):
    # Create a unique log file for playing/testing
    log_filename = f'run_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logger = logging.getLogger('run_logger')
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers on repeated runs
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Run for n_episodes
    for i_episode in range(1, n_episodes + 1):
        upper_bounds = env.observation_space.high
        lower_bounds = env.observation_space.low

        obs, _ = env.reset()
        current_state = agent.discretize_state(obs, upper_bounds, lower_bounds)
        episode_reward = 0
        t = 0
        done = False

        while not done:
            env.render()
            t += 1

            # Choose action based on the current state
            action = agent.get_action(current_state, env, epsilon=0.1)

            # Take the action in the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update the episode reward
            episode_reward += reward

            # Discretize the observed state
            new_state = agent.discretize_state(obs, upper_bounds, lower_bounds)
            current_state = new_state

        # Log the episode details
        logger.info(f'Episode {i_episode}: Total Reward: {episode_reward}, Timesteps: {t}')

    env.close()
    print(f"Logs saved to: {log_filename}")





