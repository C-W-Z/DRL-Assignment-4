import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac import SAC
from dmc import make_dmc_env

def save_random_state(self):
    """Save random states for reproducibility"""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

def load_random_state(self, state):
    """Load random states for reproducibility"""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])

def train(agent: SAC, checkpoint_path: str, log_dir: str):
    env = make_dmc_env('humanoid-walk', np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    eval_env = make_dmc_env('humanoid-walk', np.random.randint(0, 1000000), flatten=True, use_pixels=False)

    # Initialize TensorBoard writer (log_dir may be updated by checkpoint)
    writer = None

    # Load checkpoint if provided
    start_episode = 0
    total_steps = 0
    if checkpoint_path:
        load_checkpoint(checkpoint_path)
        print(f"Resumed training from checkpoint: {checkpoint_path}")
        print(f"Starting from episode {start_episode}, total steps {total_steps}")

    # Initialize TensorBoard writer if not set by checkpoint
    if writer is None:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)



if __name__ == "__main__":
    agent = SAC(
        state_dim=67,
        action_dim=21,
        hidden_dim=256,
        action_bounds=(-1.0, 1.0),
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        buffer_capacity=1_000_000,
    )