import torch
from sac import SAC

agent = SAC(
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

checkpoint = torch.load("./episode_10300.pth", weights_only=False)
agent.load_state_dict(checkpoint['agent_state_dict'], load_replay_buffer=False)
torch.save(agent.policy.state_dict(), "final.pth")
