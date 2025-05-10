import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple

def _init_weights(m: nn.Module) -> None:
    """
    Initialize weights and biases for linear layers using Xavier uniform and constant initialization.

    Args:
        m (nn.Module): A PyTorch module (e.g., nn.Linear) to initialize.
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight) # Apply Xavier uniform initialization to weights for stable training
        init.constant_(m.bias, 0)      # Set biases to 0 to avoid initial output offsets


class Actor(nn.Module):
    """
    Policy Network for SAC that outputs a Gaussian distribution over actions.
    Uses the reparameterization trick to sample actions and compute log probabilities.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        epsilon: float = 1e-8
    ) -> None:
        """
        Initialize the Actor network with separate outputs for mean and log standard deviation.

        Args:
            state_dim (int): Dimension of the state space (e.g., 67 for humanoid-walk).
            action_dim (int): Dimension of the action space (e.g., 21 for humanoid-walk).
            hidden_dim (int): Size of hidden layers (e.g., 256).
            action_bounds (Tuple[float, float]): Action range (min, max), default (-1.0, 1.0).
            epsilon (float): Small constant to prevent numerical instability in log probability.
        """
        super(Actor, self).__init__()
        # Define the main MLP for feature extraction
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
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2    # e.g., 0.0 for [-1.0, 1.0]

        self.epsilon = epsilon                              # Small constant to prevent division by zero in log probability
        self.apply(_init_weights)                           # Apply Xavier initialization to all linear layers

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

        # Compute mean of the action distribution
        mean    = self.mean(x)                  # Shape: (batch_size, action_dim)

        # Compute log standard deviation, clamped for numerical stability
        log_std = self.log_std(x)               # Shape: (batch_size, action_dim)
        log_std = torch.clamp(log_std, -20, 2)  # Restrict log_std to [-20, 2]

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the Gaussian policy using the reparameterization trick.
        Also compute the log probability of the sampled actions.

        Args:
            state (torch.Tensor): State tensor, shape (batch_size, state_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - action: Sampled action, shape (batch_size, action_dim).
                - log_prob: Log probability of the action, shape (batch_size, 1).
        """
        # Get mean and log standard deviation
        mean, log_std   = self.forward(state)
        std             = log_std.exp()         # Shape: (batch_size, action_dim)

        normal          = torch.distributions.Normal(mean, std)
        # Sample using reparameterization trick: x_t = mean + std * N(0, 1)
        x_t             = normal.rsample()      # Shape: (batch_size, action_dim)
        # Apply tanh to squash output to [-1, 1]
        y_t             = torch.tanh(x_t)       # Shape: (batch_size, action_dim)

        # Scale and shift to match action bounds
        action          = y_t * self.action_scale + self.action_bias  # Shape: (batch_size, action_dim)

        # Compute log probability of the sample
        log_prob        = normal.log_prob(x_t)  # Shape: (batch_size, action_dim)
        # Adjust log probability for tanh transformation (Jacobian correction)
        log_prob       -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        # Sum log probabilities across action dimensions
        log_prob        = log_prob.sum(dim=-1, keepdim=True)  # Shape: (batch_size, 1)
        return action, log_prob

class Critic(nn.Module):
    """
    Q-Network for SAC that predicts Q-values for state-action pairs.
    Used to estimate the expected cumulative reward Q(s, a) in continuous action spaces.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        """
        Initialize the Critic network with a sequential MLP architecture.

        Args:
            state_dim (int): Dimension of the state space (e.g., 67 for humanoid-walk).
            action_dim (int): Dimension of the action space (e.g., 21 for humanoid-walk).
            hidden_dim (int): Size of hidden layers (e.g., 256).
        """
        super(Critic, self).__init__()
        # Define the MLP: input is concatenated state and action, output is a single Q-value
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),  # First layer: processes concatenated state and action
            nn.ReLU(),                                      # Non-linearity for feature extraction
            nn.Linear(hidden_dim, hidden_dim),              # Second layer: further processes features
            nn.ReLU(),                                      # Non-linearity
            nn.Linear(hidden_dim, 1)                        # Output layer: produces a single Q-value
        )

        self.apply(_init_weights)                           # Apply Xavier initialization to all linear layers

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for given state-action pairs.

        Args:
            state (torch.Tensor): State tensor, shape (batch_size, state_dim).
            action (torch.Tensor): Action tensor, shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Q-values, shape (batch_size, 1).
        """
        # Concatenate state and action along the last dimension
        x = torch.cat([state, action], dim=-1)  # Shape: (batch_size, state_dim + action_dim)
        # Pass through the network to get Q-value
        return self.network(x)  # Shape: (batch_size, 1)
