import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, lower_bound, upper_bound, device):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.lower_bound = torch.tensor(lower_bound, dtype=torch.float32, device=device)
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.float32, device=device)

    def forward(self, state):
        action_mean = self.actor(state)
        action_mean = (action_mean + 1) * (self.upper_bound - self.lower_bound) / 2 + self.lower_bound
        value = self.critic(state)
        return action_mean, value

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, lower_bound, upper_bound, learning_rate, clip_range, value_coeff, entropy_coeff, initial_std, max_grad_norm):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim, lower_bound, upper_bound, self.device)
        self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(initial_std)), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        mean, value = self.model(state)
        std = torch.exp(self.log_std).clamp(1e-6, 50.0)
        dist = Normal(mean, std)
        return dist, value

    def get_action(self, state):
        with torch.no_grad():
            dist, value = self.forward(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], log_prob.item(), value.item()

    def get_values(self, state):
        with torch.no_grad():
            _, value = self.forward(state)
        return value.item()

    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch):
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        log_prob_batch = log_prob_batch.to(self.device)
        advantage_batch = advantage_batch.to(self.device)
        return_batch = return_batch.to(self.device)

        dist, value = self.forward(obs_batch)
        new_log_prob = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_prob - log_prob_batch)
        surr1 = ratio * advantage_batch
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_batch
        pi_loss = -torch.min(surr1, surr2).mean()
        value_loss = self.value_coeff * nn.MSELoss()(value, return_batch)  # Removed squeeze()
        entropy_loss = -self.entropy_coeff * entropy
        total_loss = pi_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        approx_kl = ((ratio - 1) - (new_log_prob - log_prob_batch)).mean()
        return pi_loss, value_loss, total_loss, approx_kl, torch.exp(self.log_std)
