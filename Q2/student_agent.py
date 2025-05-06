import gymnasium
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import os
from train_ppo import Actor, RunningStat

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim=5, action_dim=1, lower_bound=-1.0, upper_bound=1.0, device=self.device).to(self.device)

        checkpoint = torch.load("checkpoint.pt", weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        mean = checkpoint['running_stat']['mean']
        std = checkpoint['running_stat']['std']
        count = checkpoint['running_stat']['count']
        M2 = checkpoint['running_stat']['M2']

        self.running_stat = RunningStat((5,), mean, std, count, M2)

    def act(self, observation):
        # Normalize observation
        observation = np.array(observation, dtype=np.float32)
        normalized_obs = self.running_stat.normalize(observation)
        obs_tensor = torch.FloatTensor(normalized_obs).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs_tensor)
            action = action.cpu().numpy()[0]

        action = np.clip(action, -1.0, 1.0)
        return action
