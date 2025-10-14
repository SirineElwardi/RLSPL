
BUFFER_SIZE =1000000         # replay buffer size
BATCH_SIZE =50          # minibatch size
GAMMA = 50              # discount factor
TAU = 1e-3              # for soft update of target parameters

LEARNING_RATE = 50     
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic

WEIGHT_DECAY = 0        # L2 weight decay
LEARNING_PERIOD = 20    # learning frequency  

TARGET_UPDATE = 10

num_episodes = 10
print_every = 10       
hidden_dim = 16        ## 64 ## 16
Alpha= 0.2

EPSILON =50             # epsilon noise parameter
EPSILON_DECAY = 50      # decay parameter of epsilon
min_eps = 0.01
max_eps_episode = 50




