import torch
from train_ppo import Actor, RunningStat

class Agent(object):
    """PPO Agent for Pendulum-v1."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim=3, action_dim=1, lower_bound=-2.0, upper_bound=2.0, device=self.device).to(self.device)

        checkpoint = torch.load("final.pt", weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        mean = checkpoint['running_stat']['mean']
        std = checkpoint['running_stat']['std']
        count = checkpoint['running_stat']['count']

        self.running_stat = RunningStat((3,), mean, std, count)

    def act(self, observation):
        with torch.no_grad():
            state = self.running_stat.normalize(observation)
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action = self.actor(state)
            action = action.cpu().numpy()[0]
        return action
