import numpy as np
import random
import copy
import math

from DDPGModel import Actor, Critic
from DQNModel import QNetwork
from SACmodel import QNetwork as QNetworkSAC, GaussianPolicy, DeterministicPolicy
import hyperparameters
from replay_buffer import ReplayMemory, Transition
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

# ---------------------- UTILS ---------------------- #
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def hard_update(local, target):
    for target_param, param in zip(target.parameters(), local.parameters()):
        target_param.data.copy_(param.data)

# ---------------------- AGENT CLASS ---------------------- #
class Agent(object):

    # ------- DQN INIT -------
    def __init__(self, n_states, n_actions):
        self.q_local = QNetwork(n_states, n_actions, hyperparameters.hidden_dim).to(device)
        self.q_target = QNetwork(n_states, n_actions, hyperparameters.hidden_dim).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_memory = ReplayMemory(10000)
        self.optim = optim.Adam(self.q_local.parameters(), lr=hyperparameters.LEARNING_RATE)

    # ------- DDPG INIT -------
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = hyperparameters.EPSILON
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayMemory(action_size, hyperparameters.BUFFER_SIZE, hyperparameters.BATCH_SIZE, random_seed)
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyperparameters.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyperparameters.LR_CRITIC, weight_decay=hyperparameters.WEIGHT_DECAY)

    # ------- SAC INIT -------
    def __init__(self, num_inputs, action_space, device):
        self.gamma = hyperparameters.GAMMA
        self.tau = hyperparameters.TAU
        self.alpha = hyperparameters.Alpha
        self.hidden_dim = hyperparameters.hidden_dim
        self.device = device
        self.LR = hyperparameters.LEARNING_RATE
        self.target_update = hyperparameters.TARGET_UPDATE

        self.critic = QNetworkSAC(num_inputs, action_space.shape[0], self.hidden_dim).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.LR)
        self.critic_target = QNetworkSAC(num_inputs, action_space.shape[0], self.hidden_dim).to(device)
        hard_update(self.critic_target, self.critic)

        self.automatic_entropy_tuning = False
        self.policy_type = "Gaussian"
        self.replay_memory = ReplayMemory(hyperparameters.BUFFER_SIZE)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.alpha_optim = optim.Adam([self.log_alpha], lr=self.LR)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], self.hidden_dim, action_space).to(device)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.LR)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], self.hidden_dim, action_space).to(device)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.LR)
            self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=self.LR)


    # ------- Q-LEARNING INIT -------
    def __init__(self, buckets, num_episodes, min_epsilon, discount, decay, learning_rate, print_every, env):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.Q_table = np.zeros(self.buckets + (env.action_space.n,))

    # ------- Q-LEARNING Methods -------
    def update_q(self, Q_table, state, action, reward, new_state, learning_rate, discount):
        Q_table[state][action] += learning_rate * (reward + discount * np.max(Q_table[new_state]) - Q_table[state][action])

    def discretize_state(self, obs, upper_bounds, lower_bounds):
        discretized = []
        for i in range(len(obs)):
            scaling = (obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def get_action(self, state, env, epsilon):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return np.argmax(self.Q_table[state])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    # ------- SAC ACTION -------
    def get_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    # ------- DQN ACTION (epsilon-greedy) -------
    def get_action(self, state, eps, check_eps=True):
        sample = random.random()
        if not check_eps or sample > eps:
            with torch.no_grad():
                return self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
        return torch.tensor([[random.randrange(self.n_actions)]], device=device)

    # ------- DDPG ACTION (Ornstein-Uhlenbeck) -------
    def get_action(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return action

    def reset(self):
        self.noise.reset()

    # ------- DQN LEARN -------
    def learn(self, gamma):
        gamma = hyperparameters.GAMMA
        if len(self.replay_memory.memory) < hyperparameters.BATCH_SIZE:
            return
        transitions = self.replay_memory.sample(hyperparameters.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        Q_expected = self.q_local(states).gather(1, actions)
        Q_targets_next = self.q_target(next_states).detach().max(1)[0]
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        self.q_local.train()
        self.optim.zero_grad()
        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))
        loss.backward()
        self.optim.step()

    # ------- DDPG LEARN -------
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.critic_local, self.critic_target, hyperparameters.TAU)
        soft_update(self.actor_local, self.actor_target, hyperparameters.TAU)
        self.epsilon -= hyperparameters.EPSILON_DECAY
        self.noise.reset()

    def step(self, state, action, reward, next_state, done, timestep):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > hyperparameters.BATCH_SIZE and timestep % hyperparameters.LEARNING_PERIOD == 0:
            for _ in range(hyperparameters.TARGET_UPDATE):
                experiences = self.memory.sample()
                self.learn(experiences, hyperparameters.GAMMA)

    # ------- SAC LEARN -------
    def learn(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

# ------------------ NOISE ------------------ #
class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state