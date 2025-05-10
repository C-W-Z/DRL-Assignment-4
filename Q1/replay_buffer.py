import numpy as np

class ReplayBuffer:
    """高效的經驗回放緩衝區，專為一維 state 和 action 向量優化"""
    def __init__(self, capacity=1000000, state_dim=3, action_dim=1, seed=None):
        self.capacity = capacity
        self.state_dim = state_dim  # state 向量的維度
        self.action_dim = action_dim  # action 向量的維度
        self.rng = np.random.default_rng(seed)  # NumPy 隨機數生成器

        # 預分配固定形狀的 NumPy 陣列
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """將經驗存入緩衝區"""
        idx = self.pos % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """隨機採樣 batch_size 個經驗"""
        indices = self.rng.choice(self.size, size=batch_size, replace=True)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        """返回緩衝區當前大小"""
        return self.size
