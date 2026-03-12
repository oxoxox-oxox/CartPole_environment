import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from net_model.Qnet import Qnet


class DQN:
    """DQN Algorithm"""

    def __init__(self, state_dim,  hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)

        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)

        # using Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = self.target_update
        self.count = 0
        self.divice = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor
