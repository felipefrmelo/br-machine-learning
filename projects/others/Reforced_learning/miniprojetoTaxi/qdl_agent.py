import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QDl_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, gamma = 0.99, alpha = 5e-4):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.alpha = alpha
        self.seed = random.seed(seed)
        self.episode = 0
        # Q-Network
        self.qnetwork = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=alpha)

    def select_action(self, state, eps=.001):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = self.pre_process(state)
        
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def pre_process(self,state):
        return torch.tensor([state])
    
    def step(self, state, action, reward, next_state, done):
        
        self.learn(state, action, reward, next_state, done)
        
    def learn(self, state, action, reward, next_state, done):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            gamma (float): discount factor
        """
        state = self.pre_process(state)
        next_state = self.pre_process(next_state)
        
        Q_state = self.qnetwork(state).flatten()[action]
        
        TD_target = reward + self.gamma * self.qnetwork(next_state).detach().flatten().max()
        
        # Compute loss
        loss = F.mse_loss(Q_state, TD_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()       
        

