import gymnasium as gym
import numpy as np
import torch
from train import ActorCritic

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """PPO Agent for Pendulum-v1."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim=3, action_dim=1).to(self.device)
        # 載入訓練好的模型參數
        self.model.load_state_dict(torch.load("ppo_pendulum.pth", weights_only=False))
        self.model.eval()

    def act(self, observation):
        with torch.no_grad():
            state = torch.FloatTensor(observation).to(self.device)
            mean, _, _ = self.model(state.unsqueeze(0))
            action = mean.cpu().numpy()[0]
        return action
