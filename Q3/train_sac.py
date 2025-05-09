import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import random
import datetime
from collections import deque
from torch.distributions import Normal
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

###########################################
# Neural Network Models
###########################################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, lower_bound, upper_bound, hidden_dim=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.lower_bound = torch.tensor(lower_bound, dtype=torch.float32, device=self.device)
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.float32, device=self.device)
        self.to(self.device)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

    def sample(self, state):
        dist = self.forward(state)
        u = dist.rsample()
        action = torch.tanh(u)
        action = action * (self.upper_bound - self.lower_bound) / 2 + (self.upper_bound + self.lower_bound) / 2
        log_prob = dist.log_prob(u) - torch.log(1 - torch.tanh(u)**2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.forward(state)
            if deterministic:
                action = torch.tanh(dist.mean)
                action = action * (self.upper_bound - self.lower_bound) / 2 + (self.upper_bound + self.lower_bound) / 2
            else:
                action, _ = self.sample(state)
        return action.cpu().numpy()[0]

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

###########################################
# Replay Buffer
###########################################

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, seed=None):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, size=batch_size, replace=True)
        return {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }

    def __len__(self):
        return self.size

###########################################
# Utility Functions
###########################################

class RunningStat:
    def __init__(self, shape, mean=None, std=None, count=0):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32) if mean is None else mean
        self.std = np.ones(shape, dtype=np.float32) if std is None else std
        self.count = count
        self.M2 = np.zeros(shape, dtype=np.float32)

    def update(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            variance = self.M2 / (self.count - 1)
            variance = np.maximum(variance, 0)
            self.std = np.sqrt(variance)
        else:
            self.std = np.ones(self.shape, dtype=np.float32)

    def normalize(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        return (x - self.mean) / (self.std + 1e-8)

def save_random_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

def load_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])

def save_checkpoint(policy, q1, q2, running_stat, step, mean_rewards, log_dir, checkpoint_path):
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'q1_state_dict': q1.state_dict(),
        'q2_state_dict': q2.state_dict(),
        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        'q_optimizer_state_dict': q_optimizer.state_dict(),
        'log_alpha': log_alpha,
        'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
        'step': step,
        'mean_rewards': mean_rewards,
        'random_state': save_random_state(),
        'running_stat': {
            'mean': running_stat.mean,
            'std': running_stat.std,
            'count': running_stat.count,
            'M2': running_stat.M2
        },
        'log_dir': log_dir
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(policy, q1, q2, running_stat, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, None, None
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if policy is not None:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    if q1 is not None and q2 is not None:
        q1.load_state_dict(checkpoint['q1_state_dict'])
        q2.load_state_dict(checkpoint['q2_state_dict'])
        q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
    if running_stat is not None:
        running_stat.mean = checkpoint['running_stat']['mean']
        running_stat.std = checkpoint['running_stat']['std']
        running_stat.count = checkpoint['running_stat']['count']
        running_stat.M2 = checkpoint['running_stat'].get('M2', np.zeros(running_stat.shape, dtype=np.float32))

    global log_alpha, alpha_optimizer
    log_alpha = checkpoint['log_alpha']
    alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

    if 'random_state' in checkpoint:
        load_random_state(checkpoint['random_state'])

    log_dir = checkpoint.get('log_dir', 'Log')
    return (checkpoint['step'], checkpoint['mean_rewards'], log_dir)

###########################################
# SAC Implementation
###########################################

def train_sac(env_name="humanoid-walk",
              total_timesteps=3000000,
              batch_size=256,
              replay_buffer_capacity=1000000,
              learning_rate=3e-4,
              gamma=0.99,
              tau=0.005,
              target_entropy_scale=0.2,
              reward_scale=1.0,
              start_steps=10000,
              update_frequency=1,
              updates_per_step=1,
              save_interval=5000,
              checkpoint_path="humanoid_walk_sac_checkpoint.pt"):

    global device, policy_optimizer, q_optimizer, alpha_optimizer, log_alpha
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_dmc_env(env_name, seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    policy = PolicyNetwork(state_dim, action_dim, lower_bound, upper_bound).to(device)
    q1 = QNetwork(state_dim, action_dim).to(device)
    q2 = QNetwork(state_dim, action_dim).to(device)
    target_q1 = QNetwork(state_dim, action_dim).to(device)
    target_q2 = QNetwork(state_dim, action_dim).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=learning_rate)
    log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)
    target_entropy = -target_entropy_scale * action_dim

    replay_buffer = ReplayBuffer(replay_buffer_capacity, state_dim, action_dim)
    running_stat = RunningStat((state_dim,))

    episode_count, _, log_dir = load_checkpoint(None, None, None, None, checkpoint_path)
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("Log", f"run_{timestamp}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    print(f"Logging to TensorBoard at {log_dir}")

    step, mean_rewards, _ = load_checkpoint(policy, q1, q2, running_stat, checkpoint_path) if episode_count is not None else (0, [], None)
    if step is not None:
        print(f"Resumed training from checkpoint at step={step}")

    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_rewards = []
    recent_rewards = deque(maxlen=100)

    while step < total_timesteps:
        running_stat.update(state)
        state_normalized = running_stat.normalize(state)

        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = policy.get_action(state_normalized, deterministic=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_steps += 1
        step += 1

        replay_buffer.push(state_normalized, action, reward * reward_scale, running_stat.normalize(next_state), done)

        state = next_state

        if len(replay_buffer) > batch_size and step >= start_steps:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                states = batch['states']
                actions = batch['actions']
                rewards = batch['rewards']
                next_states = batch['next_states']
                dones = batch['dones']

                with torch.no_grad():
                    next_actions, next_log_probs = policy.sample(next_states)
                    next_q1 = target_q1(next_states, next_actions)
                    next_q2 = target_q2(next_states, next_actions)
                    next_q = torch.min(next_q1, next_q2) - log_alpha.exp() * next_log_probs
                    target_q = rewards + (1 - dones) * gamma * next_q

                q1_loss = F.mse_loss(q1(states, actions), target_q)
                q2_loss = F.mse_loss(q2(states, actions), target_q)
                q_loss = (q1_loss + q2_loss) / 2

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                actions_pi, log_probs_pi = policy.sample(states)
                q1_pi = q1(states, actions_pi)
                q2_pi = q2(states, actions_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                policy_loss = (log_alpha.exp() * log_probs_pi - q_pi).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                alpha_loss = (-log_alpha * (log_probs_pi.detach() + target_entropy)).mean()

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                for target_param, param in zip(target_q1.parameters(), q1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_q2.parameters(), q2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                writer.add_scalar("train/q_loss", q_loss.item(), step)
                writer.add_scalar("train/policy_loss", policy_loss.item(), step)
                writer.add_scalar("train/alpha_loss", alpha_loss.item(), step)
                writer.add_scalar("train/alpha", log_alpha.exp().item(), step)

        if done:
            episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)
            writer.add_scalar("misc/ep_reward", episode_reward, step)
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Step {step} | Episode {len(episode_rewards)} | Reward {episode_reward:.2f} | Mean Reward {mean_reward:.2f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0

        if step % save_interval == 0 and step > 0:
            save_checkpoint(policy, q1, q2, running_stat, step, list(recent_rewards), log_dir, checkpoint_path)

        if len(recent_rewards) == 100 and np.mean(recent_rewards) > 500:
            print(f"Achieved mean reward {np.mean(recent_rewards):.2f}. Stopping training.")
            break

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.grid(True)
    plt.savefig("episode_reward.png")

    save_checkpoint(policy, q1, q2, running_stat, step, list(recent_rewards), log_dir, checkpoint_path)

    writer.close()
    env.close()

if __name__ == "__main__":
    checkpoint_path = "humanoid_walk_sac_checkpoint.pt"
    train_sac(total_timesteps=3000000, learning_rate=3e-4, checkpoint_path=checkpoint_path)
