import numpy as np
from typing import Union, Tuple, Dict, Any, Optional

import torch

class ReplayBuffer:
    """Efficient replay buffer optimized for 1D state and action vectors"""
    def __init__(
        self,
        capacity    : int ,
        state_dim   : int,
        action_dim  : int,
        device      : torch.device,
    ) -> None:
        self.device             = device
        self.capacity           = capacity
        self.state_dim          = state_dim
        self.action_dim         = action_dim
        self.count              = 0
        self.size               = 0
        self.reward_sum         = 0.0
        self.reward_square_sum  = 0.0
        self.states             = torch.zeros((capacity, state_dim ), dtype=torch.float32, device=device)
        self.actions            = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards            = torch.zeros((capacity, 1         ), dtype=torch.float32, device=device)
        self.next_states        = torch.zeros((capacity, state_dim ), dtype=torch.float32, device=device)
        self.dones              = torch.zeros((capacity, 1         ), dtype=torch.float32, device=device)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        next_state: np.ndarray,
        done: Union[float, np.ndarray]
    ) -> None:
        """
        Store a new experience in the replay buffer.
        Args:
            state (np.ndarray): Current state, shape (state_dim,).
            action (np.ndarray): Action taken, shape (action_dim,).
            reward (Union[float, np.ndarray]): Reward received, scalar or shape (1,).
            next_state (np.ndarray): Next state, shape (state_dim,).
            done (Union[float, np.ndarray]): Termination flag, scalar or shape (1,).
        """
        idx                     = self.count % self.capacity

        # Update Normalization Params
        self.count             += 1
        self.reward_sum        += reward
        self.reward_square_sum += reward ** 2
        mean                    = self.reward_sum / self.count
        var                     = max(self.reward_square_sum / self.count - mean ** 2, 1e-4)
        normalized              = (reward - mean) / np.sqrt(var)

        self.states     [idx]   = torch.tensor(state         , device=self.device)
        self.actions    [idx]   = torch.tensor(action        , device=self.device)
        self.rewards    [idx]   = torch.tensor([[normalized]], device=self.device)
        self.next_states[idx]   = torch.tensor(next_state    , device=self.device)
        self.dones      [idx]   = torch.tensor(done          , device=self.device)

        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - states: Batch of states, shape (batch_size, state_dim).
                - actions: Batch of actions, shape (batch_size, action_dim).
                - rewards: Batch of rewards, shape (batch_size, 1).
                - next_states: Batch of next states, shape (batch_size, state_dim).
                - dones: Batch of done flags, shape (batch_size, 1).
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states     [indices],
            self.actions    [indices],
            self.rewards    [indices],
            self.next_states[indices],
            self.dones      [indices]
        )

    def __len__(self) -> int:
        """
        Return the current size of the replay buffer.
        Returns:
            int: Number of experiences stored.
        """
        return self.size

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary containing the replay buffer's state for checkpointing.
        Returns:
            Dict[str, Any]: Dictionary with buffer data and random number generator state.
        """
        return {
            'size'              : self.size,
            'count'             : self.count,
            'reward_sum'        : self.reward_sum,
            'reward_square_sum' : self.reward_square_sum,
            'states'            : self.states     [:self.size],
            'actions'           : self.actions    [:self.size],
            'rewards'           : self.rewards    [:self.size],
            'next_states'       : self.next_states[:self.size],
            'dones'             : self.dones      [:self.size],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the replay buffer's state from a dictionary.
        Args:
            state_dict (Dict[str, Any]): Dictionary containing buffer data and RNG state.
        """
        self.size                       = state_dict['size']
        self.count                      = state_dict['count']
        self.reward_sum                 = state_dict['reward_sum']
        self.reward_square_sum          = state_dict['reward_square_sum']
        self.states     [:self.size]    = state_dict['states']
        self.actions    [:self.size]    = state_dict['actions']
        self.rewards    [:self.size]    = state_dict['rewards']
        self.next_states[:self.size]    = state_dict['next_states']
        self.dones      [:self.size]    = state_dict['dones']
