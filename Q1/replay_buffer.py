import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """Stores experience tuples for off-policy training"""
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Random sampling with replacement
        transitions = random.sample(self.buffer, batch_size)
        # Transpose the batch for easier access
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)
