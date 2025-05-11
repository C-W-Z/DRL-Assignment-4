import torch
from sac import SAC

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.agent = SAC(
            state_dim       = 67,
            action_dim      = 21,
            hidden_dim      = 256,
            action_bounds   = (-1.0, 1.0),
            gamma           = 0.99,
            tau             = 0.005,
            lr              = 1e-4,
            alpha           = 0.2,
            icm_eta         = 0.1,
            forward_weight  = 1.0,
            inverse_weight  = 0.1,
            buffer_capacity = 100,
        )

        self.agent.policy.load_state_dict(
            torch.load(
                "./final.pth",
                weights_only=False,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        )

    def act(self, observation):
        return self.agent.select_action(observation, evaluate=True)
