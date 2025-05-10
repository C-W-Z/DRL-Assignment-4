import argparse
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
import time

# Add parent directory to sys.path for dmc import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env(domain_name: str, task_name: str, seed: int) -> Any:
    """Create a DeepMind Control Suite environment."""
    env_name = f"{domain_name}-{task_name}"
    env = make_dmc_env(env_name, seed, flatten=True, use_pixels=False)
    return env

# Replay Buffer with reward normalization
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device: torch.device) -> None:
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

        # Statistics for normalization
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        self.count = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float) -> None:
        """Add a transition and update normalized reward."""
        # Store state, action, next_state, done
        self.state[self.ptr] = torch.tensor(state, device=self.device)
        self.action[self.ptr] = torch.tensor(action, device=self.device)
        self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
        self.done[self.ptr] = torch.tensor(done, device=self.device)

        # Update reward statistics
        self.count += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward ** 2
        mean = self.reward_sum / self.count
        var = max(self.reward_sq_sum / self.count - mean ** 2, 1e-4)
        normalized = (reward - mean) / np.sqrt(var)
        self.reward[self.ptr] = torch.tensor([[normalized]], device=self.device)

        # Increment pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx]
        )

# Actor network for SAC
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log-probability."""
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

# Critic network for SAC
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# Intrinsic Curiosity Module (ICM)
class ICM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(ICM, self).__init__()
        # Encode state
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        s_enc = self.encode(state)
        ns_enc = self.encode(next_state)
        pred = self.forward_model(torch.cat([s_enc, action], dim=-1))
        return F.mse_loss(pred, ns_enc)

    def inverse_loss(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s_enc = self.encode(state)
        ns_enc = self.encode(next_state)
        pred = self.inverse_model(torch.cat([s_enc, ns_enc], dim=-1))
        return F.mse_loss(pred, action)

    def intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        s_enc = self.encode(state)
        ns_enc = self.encode(next_state)
        pred = self.forward_model(torch.cat([s_enc, action], dim=-1))
        # Half mean square error
        return 0.5 * ((pred - ns_enc)**2).mean(dim=-1, keepdim=True)

# SAC agent with ICM
class SAC:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        print(f"Device: {self.device}")

        # Environment setup
        self.env = make_env(args.domain_name, args.task_name, args.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, args.hidden_dim, self.max_action).to(self.device)
        self.critic1 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target1 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target2 = Critic(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        # ICM
        self.icm = ICM(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.lr_critic)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.lr_critic)
        self.icm_opt = optim.Adam(self.icm.parameters(), lr=args.lr_icm)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=args.lr_alpha)
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -self.action_dim

        # Replay buffer
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, args.replay_size, self.device)

        # TensorBoard writer
        log_dir = f"runs/run_{int(time.time())}"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Output file
        self.file = open("output.txt", "w")
        self.save_path = "model.pth"

    def update(self, total_steps: int) -> None:
        if self.buffer.size < self.args.batch_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.args.batch_size)
        # Intrinsic reward
        with torch.no_grad():
            ir = self.icm.intrinsic_reward(s, a, ns)
        total_r = r + self.args.intrinsic_weight * ir

        # Critic update
        with torch.no_grad():
            na, lp = self.actor.sample(ns)
            q1n = self.target1(ns, na)
            q2n = self.target2(ns, na)
            qn = torch.min(q1n, q2n) - self.alpha * lp
            target_q = total_r + (1 - d) * self.args.gamma * qn

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        loss1 = F.mse_loss(q1, target_q)
        loss2 = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        loss1.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        loss2.backward()
        self.critic2_opt.step()

        # Actor update
        na, lp = self.actor.sample(s)
        q1p = self.critic1(s, na)
        q2p = self.critic2(s, na)
        qp = torch.min(q1p, q2p)
        actor_loss = (self.alpha * lp - qp).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        # ICM update
        f_loss = self.icm.forward_loss(s, a, ns)
        i_loss = self.icm.inverse_loss(s, ns, a)
        icm_loss = self.args.forward_weight * f_loss + self.args.inverse_weight * i_loss
        self.icm_opt.zero_grad()
        icm_loss.backward()
        self.icm_opt.step()

        # Log to TensorBoard
        self.writer.add_scalar('Losses/Policy', actor_loss.item(), total_steps)
        self.writer.add_scalar('Losses/Q1', loss1.item(), total_steps)
        self.writer.add_scalar('Losses/Q2', loss2.item(), total_steps)
        self.writer.add_scalar('Losses/Forward', f_loss.item(), total_steps)
        self.writer.add_scalar('Losses/Inverse', i_loss.item(), total_steps)
        self.writer.add_scalar('Rewards/Intrinsic', ir.mean().item(), total_steps)
        self.writer.add_scalar('Misc/Alpha', self.alpha.item(), total_steps)

        # Target critic soft update
        for tp, p in zip(self.target1.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.args.tau * p.data + (1 - self.args.tau) * tp.data)
        for tp, p in zip(self.target2.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.args.tau * p.data + (1 - self.args.tau) * tp.data)

    def train(self) -> None:
        rewards = deque(maxlen=100)
        total_steps = 0
        pbar = tqdm(range(self.args.max_episodes), desc="Training")
        for ep in pbar:
            obs, _ = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device)
            ep_reward = 0
            for t in range(self.args.max_steps):
                with torch.no_grad():
                    action, _ = self.actor.sample(state.unsqueeze(0))
                a = action.cpu().numpy().flatten()
                next_obs, rew, done, term, _ = self.env.step(a)
                ep_reward += rew if rew is not None else 0.0
                done_flag = float(done or term)
                self.buffer.add(state.cpu().numpy(), a, rew or 0.0,
                                np.array(next_obs), done_flag)
                state = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                # Update networks
                for _ in range(self.args.update_steps):
                    self.update(total_steps)
                    total_steps += 1
                if done_flag:
                    break
            rewards.append(ep_reward)
            avg = np.mean(rewards)
            # Log episode rewards to TensorBoard
            self.writer.add_scalar('Rewards/Episode', ep_reward, ep)
            self.writer.add_scalar('Rewards/Avg100Episodes', avg, ep)
            # Write to file
            self.file.write(f"Episode {ep+1}\tReward {ep_reward:.2f}\tAvg100 {avg:.2f}\n")
            self.file.flush()
            # Save checkpoint
            if (ep+1) % self.args.save_interval == 0:
                torch.save(self.actor.state_dict(), self.save_path)
            pbar.set_description(f"Ep {ep+1}, Reward: {ep_reward:.2f}, Avg100: {avg:.2f}")
        # Final save
        torch.save(self.actor.state_dict(), self.save_path)
        # Close TensorBoard writer and output file
        self.writer.close()
        self.file.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, default="humanoid")
    parser.add_argument("--task_name", type=str, default="walk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--lr_alpha", type=float, default=1e-4)
    parser.add_argument("--lr_icm", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--replay_size", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--update_steps", type=int, default=1)
    parser.add_argument("--intrinsic_weight", type=float, default=0.1)
    parser.add_argument("--forward_weight", type=float, default=1.0)
    parser.add_argument("--inverse_weight", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sac = SAC(args)
    sac.train()