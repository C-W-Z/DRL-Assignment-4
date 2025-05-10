import numpy as np

class ReplayBuffer:
    """Efficient replay buffer optimized for 1D state and action vectors"""
    def __init__(self, capacity=1_000_000, state_dim=67, action_dim=21, seed=None):
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

    def push(self, state, action, reward, next_state, done):
        idx                     = self.pos % self.capacity
        self.states[idx]        = state
        self.actions[idx]       = action
        self.rewards[idx]       = reward
        self.next_states[idx]   = next_state
        self.dones[idx]         = done
        self.pos               += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, size=batch_size, replace=True)
        return (
            self.states     [indices],
            self.actions    [indices],
            self.rewards    [indices],
            self.next_states[indices],
            self.dones      [indices]
        )

    def __len__(self):
        return self.size

    def state_dict(self):
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

    def load_state_dict(self, state_dict):
        self.size                       = state_dict['size']
        self.pos                        = state_dict['pos']
        self.states     [:self.size]    = state_dict['states']
        self.actions    [:self.size]    = state_dict['actions']
        self.rewards    [:self.size]    = state_dict['rewards']
        self.next_states[:self.size]    = state_dict['next_states']
        self.dones      [:self.size]    = state_dict['dones']
        self.rng.__setstate__(state_dict['rng'])
