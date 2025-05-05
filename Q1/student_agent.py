import gymnasium as gym
import numpy as np
import torch
from Policy import ActorCritic

class RunningStat:
    def __init__(self, shape, mean, std, count):
        self.mean = mean
        self.std = std
        self.count = count
        self.shape = shape

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class Agent(object):
    """PPO Agent for Pendulum-v1."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim=3, action_dim=1, lower_bound=-2.0, upper_bound=2.0, device=self.device).to(self.device)
        self.model.actor.load_state_dict(torch.load("./models/ppo_actor.pth", weights_only=False))
        self.model.eval()
        # 載入正則化參數
        mean = np.load("./models/running_stat_mean.npy")
        std = np.load("./models/running_stat_std.npy")
        count = np.load("./models/running_stat_count.npy")
        self.running_stat = RunningStat((3,), mean, std, count)

    def act(self, observation):
        with torch.no_grad():
            state = self.running_stat.normalize(observation)
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action, _ = self.model(state)
            action = action.cpu().numpy()[0]
        return action
