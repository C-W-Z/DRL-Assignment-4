import numpy as np

class ReplayBuffer:
    """Efficient replay buffer optimized for 1D state and action vectors"""
    def __init__(self, capacity=1000000, state_dim=67, action_dim=21, seed=None):  # 更新維度
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        idx = self.pos % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, size=batch_size, replace=True)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size
