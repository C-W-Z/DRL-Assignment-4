import numpy as np
from typing import Union, Tuple, Dict, Any, Optional

class ReplayBuffer:
    """Efficient replay buffer optimized for 1D state and action vectors"""
    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = 67,
        action_dim: int = 21,
        seed: Optional[Union[int, np.random.SeedSequence]] = None
    ) -> None:
        self.capacity       = capacity
        self.state_dim      = state_dim
        self.action_dim     = action_dim
        self.pos            = 0
        self.size           = 0
        self.rng            = np.random.default_rng(seed)
        self.states         = np.zeros((capacity, state_dim ), dtype=np.float32)
        self.actions        = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards        = np.zeros((capacity, 1         ), dtype=np.float32)
        self.next_states    = np.zeros((capacity, state_dim ), dtype=np.float32)
        self.dones          = np.zeros((capacity, 1         ), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        next_state: np.ndarray,
        done: Union[bool, np.ndarray]
    ) -> None:
        """
        Store a new experience in the replay buffer.
        Args:
            state (np.ndarray): Current state, shape (state_dim,).
            action (np.ndarray): Action taken, shape (action_dim,).
            reward (Union[float, np.ndarray]): Reward received, scalar or shape (1,).
            next_state (np.ndarray): Next state, shape (state_dim,).
            done (Union[bool, np.ndarray]): Termination flag, scalar or shape (1,).
        """
        idx                     = self.pos % self.capacity
        self.states     [idx]   = state
        self.actions    [idx]   = action
        self.rewards    [idx]   = reward
        self.next_states[idx]   = next_state
        self.dones      [idx]   = done
        self.pos               += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly sample a batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - states: Batch of states, shape (batch_size, state_dim).
                - actions: Batch of actions, shape (batch_size, action_dim).
                - rewards: Batch of rewards, shape (batch_size, 1).
                - next_states: Batch of next states, shape (batch_size, state_dim).
                - dones: Batch of done flags, shape (batch_size, 1).
        """
        indices = self.rng.choice(self.size, size=batch_size, replace=True)
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
            'size'          : self.size,
            'pos'           : self.pos,
            'states'        : self.states     [:self.size],
            'actions'       : self.actions    [:self.size],
            'rewards'       : self.rewards    [:self.size],
            'next_states'   : self.next_states[:self.size],
            'dones'         : self.dones      [:self.size],
            'rng'           : self.rng.__getstate__(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the replay buffer's state from a dictionary.
        Args:
            state_dict (Dict[str, Any]): Dictionary containing buffer data and RNG state.
        """
        self.size                       = state_dict['size']
        self.pos                        = state_dict['pos']
        self.states     [:self.size]    = state_dict['states']
        self.actions    [:self.size]    = state_dict['actions']
        self.rewards    [:self.size]    = state_dict['rewards']
        self.next_states[:self.size]    = state_dict['next_states']
        self.dones      [:self.size]    = state_dict['dones']
        self.rng.__setstate__(state_dict['rng'])
