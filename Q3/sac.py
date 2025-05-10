
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from typing import Tuple, Dict, Any

def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(
        self,
        state_dim       : int,
        action_dim      : int,
        hidden_dim      : int                 = 256,
        action_bounds   : Tuple[float, float] = (-1.0, 1.0),
        epsilon         : float               = 1e-6
    ) -> None:
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),               # First layer: processes state input
            nn.ReLU(),                                      # Non-linearity
            nn.Linear(hidden_dim, hidden_dim),              # Second layer: further processes features
            nn.ReLU(),                                      # Non-linearity
        )
        self.mean = nn.Linear(hidden_dim, action_dim)       # Output layer for mean of the Gaussian distribution
        self.log_std = nn.Linear(hidden_dim, action_dim)    # Output layer for log standard deviation of the Gaussian distribution
        # Compute scaling factors to map tanh output to action bounds
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2   # e.g., 1.0 for [-1.0, 1.0]
        self.action_bias  = (action_bounds[1] + action_bounds[0]) / 2   # e.g., 0.0 for [-1.0, 1.0]
        self.epsilon      = epsilon
        self.apply(_init_weights)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and log standard deviation of the Gaussian policy.
        Args:
            state (torch.Tensor): State tensor, shape (batch_size, state_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mean: Mean of the Gaussian, shape (batch_size, action_dim).
                - log_std: Log standard deviation, shape (batch_size, action_dim).
        """
        x       = self.network(state)           # Shape: (batch_size, hidden_dim)
        mean    = self.mean(x)                  # Shape: (batch_size, action_dim)
        log_std = self.log_std(x)               # Shape: (batch_size, action_dim)
        log_std = torch.clamp(log_std, -20, 2)  # Clamped for numerical stability
        return mean, log_std

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic action.
        Args:
            state (torch.Tensor): State tensor, shape (batch_size, state_dim).
        Returns:
            action (torch.Tensor): shape (batch_size, action_dim)
        """
        mean, _ = self.forward(state)
        action  = torch.tanh(mean)  # Shape: (batch_size, action_dim)
        action  = action * self.action_scale + self.action_bias
        return action               # Shape: (batch_size, action_dim)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the Gaussian policy using the reparameterization trick.
        Also compute the log probability of the sampled actions
        Args:
            state (torch.Tensor): State tensor, shape (batch_size, state_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - action: Sampled action, shape (batch_size, action_dim).
                - log_prob: Log probability of the action, shape (batch_size, 1).
        """
        mean, log_std   = self.forward(state)
        std             = log_std.exp()                         # Shape: (batch_size, action_dim)
        dist            = torch.distributions.Normal(mean, std)
        x_t             = dist.rsample()                        # Shape: (batch_size, action_dim)
        y_t             = torch.tanh(x_t)                       # Shape: (batch_size, action_dim)
        action          = y_t * self.action_scale + self.action_bias  # Shape: (batch_size, action_dim)
        log_prob        = dist.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon) # Shape: (batch_size, action_dim)
        log_prob        = log_prob.sum(dim=-1, keepdim=True)    # Shape: (batch_size, 1)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),  # First layer: processes concatenated state and action
            nn.ReLU(),                                      # Non-linearity for feature extraction
            nn.Linear(hidden_dim, hidden_dim),              # Second layer: further processes features
            nn.ReLU(),                                      # Non-linearity
            nn.Linear(hidden_dim, 1)                        # Output layer: produces a single Q-value
        )
        self.apply(_init_weights)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for given state-action pairs.
        Args:
            state (torch.Tensor): State tensor, shape (batch_size, state_dim).
            action (torch.Tensor): Action tensor, shape (batch_size, action_dim).
        Returns:
            torch.Tensor: Q-values, shape (batch_size, 1).
        """
        x = torch.cat([state, action], dim=-1)  # Shape: (batch_size, state_dim + action_dim)
        return self.network(x)                  # Shape: (batch_size, 1)

class ICM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_dim)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, hidden_dim)
        )

        self.apply(_init_weights)

    def forward(
        self,
        state       : torch.Tensor,
        action      : torch.Tensor,
        next_state  : torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute encoded features, predicted next features, predicted action, and intrinsic reward.
        Args:
            state (torch.Tensor): Current state, shape (batch_size, state_dim).
            action (torch.Tensor): Action taken, shape (batch_size, action_dim).
            next_state (torch.Tensor): Next state, shape (batch_size, state_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - intrinsic_reward: Intrinsic reward based on forward model error, shape (batch_size, 1).
                - forward_loss: Forward model prediction loss, shape (batch_size, 1).
                - inverse_loss: Inverse model prediction loss, shape (batch_size, 1).
        """
        # Encode current and next states
        state_features      = self.encoder(state)                                   # Shape: (batch_size, feature_dim)
        next_state_features = self.encoder(next_state)                              # Shape: (batch_size, feature_dim)

        # Forward model: Predict next state features
        forward_input           = torch.cat([state_features, action], dim=-1)       # Shape: (batch_size, feature_dim + action_dim)
        predicted_next_features = self.forward_model(forward_input)                 # Shape: (batch_size, feature_dim)

        # Inverse model: Predict action
        inverse_input    = torch.cat([state_features, next_state_features], dim=-1) # Shape: (batch_size, feature_dim * 2)
        predicted_action = self.inverse_model(inverse_input)                        # Shape: (batch_size, action_dim)

        with torch.no_grad():
            # Compute intrinsic reward (forward model prediction error)
            intrinsic_reward = 0.5 * (next_state_features - predicted_next_features).pow(2).mean(dim=-1, keepdim=True)

        # Compute losses
        forward_loss = F.mse_loss(predicted_next_features, next_state_features)
        inverse_loss = F.mse_loss(predicted_action, action)

        return intrinsic_reward, forward_loss, inverse_loss

class SAC:
    """Soft Actor-Critic for continuous action spaces"""
    def __init__(
        self,
        state_dim       : int,
        action_dim      : int,
        hidden_dim      : int                   = 256,
        action_bounds   : Tuple[float, float]   = (-1.0, 1.0),
        lr              : float                 = 2.5e-4,
        gamma           : float                 = 0.99,
        tau             : float                 = 0.005,
        alpha           : float                 = 0.2,
        icm_eta         : float                 = 0.1,
        icm_beta         : float                = 0.2,
        buffer_capacity : int                   = 1_000_000,
        device          : torch.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        self.device = device
        self.gamma  = gamma
        self.tau    = tau
        self.alpha  = alpha

        # Networks
        self.policy = Actor (state_dim, action_dim, hidden_dim, action_bounds).to(device)
        self.q1     = Critic(state_dim, action_dim, hidden_dim)               .to(device)
        self.q2     = Critic(state_dim, action_dim, hidden_dim)               .to(device)

        # Target networks
        self.q1_target = Critic(state_dim, action_dim, hidden_dim)            .to(device)
        self.q2_target = Critic(state_dim, action_dim, hidden_dim)            .to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer   = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer       = optim.Adam(self.q1.parameters()    , lr=lr)
        self.q2_optimizer       = optim.Adam(self.q2.parameters()    , lr=lr)

        # Automatic entropy tuning
        self.target_entropy     = -action_dim  # Target entropy = -|A|
        self.log_alpha          = torch.zeros(1, requires_grad=True  , device=device)
        self.alpha_optimizer    = optim.Adam([self.log_alpha]        , lr=lr)

        # ICM
        self.icm            = ICM(state_dim, action_dim, hidden_dim)          .to(device)
        self.icm_optimizer  = optim.Adam(self.icm.parameters()       , lr=lr)
        self.icm_eta        = icm_eta
        self.icm_beta       = icm_beta

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim, action_dim=action_dim)

    def select_action(self, state: torch.Tensor, evaluate: bool=False) -> float:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate: action    = self.policy.act(state)
            else:        action, _ = self.policy.sample(state)
            return action.cpu().numpy()[0]

    def soft_update(self) -> None:
        for target_p, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)
        for target_p, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)

    def learn(self, batch_size: int) -> None:
        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.sample(batch_size)

        state_b         = torch.FloatTensor(state_b)     .to(self.device)
        action_b        = torch.FloatTensor(action_b)    .to(self.device)
        reward_b        = torch.FloatTensor(reward_b)    .to(self.device)
        next_state_b    = torch.FloatTensor(next_state_b).to(self.device)
        done_b          = torch.FloatTensor(done_b)      .to(self.device)

        intrinsic_reward, forward_loss, inverse_loss = self.icm(state_b, action_b, next_state_b)

        # Critic Learn
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state_b)
            q1_next         = self.q1_target(next_state_b, next_actions)
            q2_next         = self.q2_target(next_state_b, next_actions)
            q_next          = torch.min(q1_next, q2_next)
            value_target    = q_next - self.alpha * next_log_probs
            total_reward    = reward_b + self.icm_eta * intrinsic_reward
            q_target        = total_reward + (1 - done_b) * self.gamma * value_target

        q1_pred = self.q1(state_b, action_b)
        q2_pred = self.q2(state_b, action_b)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Actor Learn
        new_actions, log_probs = self.policy.sample(state_b)
        q1_new  = self.q1(state_b, new_actions)
        q2_new  = self.q2(state_b, new_actions)
        q_new   = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Entropy Tuning
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # ICM Learn
        icm_loss = (1 - self.icm_beta) * inverse_loss + self.icm_beta * forward_loss
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        self.soft_update()

        return policy_loss.item(), q1_loss.item(), q2_loss.item(), forward_loss.item(), inverse_loss.item(), intrinsic_reward.mean().item()

    def state_dict(self, replay_buffer: bool=True) -> Dict[str, Any]:
        state_dict = {
            'policy_state_dict'             : self.policy.state_dict(),
            'q1_state_dict'                 : self.q1.state_dict(),
            'q2_state_dict'                 : self.q2.state_dict(),
            'q1_target_state_dict'          : self.q1_target.state_dict(),
            'q2_target_state_dict'          : self.q2_target.state_dict(),
            'policy_optimizer_state_dict'   : self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict'       : self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict'       : self.q2_optimizer.state_dict(),
            'alpha'                         : self.alpha,
            'log_alpha'                     : self.log_alpha,
            'alpha_optimizer_state_dict'    : self.alpha_optimizer.state_dict(),
            'icm'                           : self.icm.state_dict(),
            'icm_optimizer'                 : self.icm_optimizer.state_dict(),
        }
        if replay_buffer:
            state_dict['replay_buffer'] = self.replay_buffer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], load_replay_buffer: bool=True) -> None:
        self.policy             .load_state_dict(state_dict['policy_state_dict'])
        self.q1                 .load_state_dict(state_dict['q1_state_dict'])
        self.q2                 .load_state_dict(state_dict['q2_state_dict'])
        self.q1_target          .load_state_dict(state_dict['q1_target_state_dict'])
        self.q2_target          .load_state_dict(state_dict['q2_target_state_dict'])
        self.policy_optimizer   .load_state_dict(state_dict['policy_optimizer_state_dict'])
        self.q1_optimizer       .load_state_dict(state_dict['q1_optimizer_state_dict'])
        self.q2_optimizer       .load_state_dict(state_dict['q2_optimizer_state_dict'])
        self.alpha              = state_dict['alpha']
        self.log_alpha          = state_dict['log_alpha']
        self.alpha_optimizer    .load_state_dict(state_dict['alpha_optimizer_state_dict'])
        self.icm                .load_state_dict(state_dict['icm'])
        self.icm_optimizer      .load_state_dict(state_dict['icm_optimizer'])
        if load_replay_buffer and 'replay_buffer' in state_dict:
            self.replay_buffer  .load_state_dict(state_dict['replay_buffer'])
