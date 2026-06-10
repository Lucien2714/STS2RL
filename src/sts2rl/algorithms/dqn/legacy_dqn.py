"""Legacy standalone DQN components kept for reference."""

import torch
from torch import nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """Ring-buffer replay memory for legacy DQN experiments."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state,action,next_state,reward):
        """Store one transition, overwriting the oldest when full."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state,action,next_state,reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.memory)
class DQN:
    """Three-layer feed-forward Q-network used by legacy experiments."""

    def __init__(self, n_states, n_actions):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)    
    def forward(self, x):
        """Return Q-values for each action."""
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        
