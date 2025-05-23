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
from torch.distributions import Normal

###########################################
# Neural Network Models
###########################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lower_bound, upper_bound, device):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.lower_bound = torch.tensor(lower_bound, dtype=torch.float32, device=device)
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.float32, device=device)

    def forward(self, state):
        action_mean = self.network(state)
        action_mean = (action_mean + 1) * (self.upper_bound - self.lower_bound) / 2 + self.lower_bound
        return action_mean

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        value = self.network(state)
        return value

###########################################
# PPO Policy Implementation
###########################################

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, lower_bound, upper_bound, learning_rate=2.5e-4,
                 clip_range=0.2, value_coeff=0.5, entropy_coeff=0.01, initial_std=1.0, max_grad_norm=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, lower_bound, upper_bound, self.device)
        self.critic = Critic(state_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(initial_std)), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        mean = self.actor(state)
        value = self.critic(state)
        std = torch.exp(self.log_std).clamp(1e-6, 50.0)
        dist = Normal(mean, std)
        return dist, value

    def get_action(self, state):
        with torch.no_grad():
            dist, value = self.forward(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], log_prob.item(), value.item()

    def get_values(self, state):
        with torch.no_grad():
            _, value = self.forward(state)
        return value.item()

    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch):
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        log_prob_batch = log_prob_batch.to(self.device)
        advantage_batch = advantage_batch.to(self.device)
        return_batch = return_batch.to(self.device)

        dist, value = self.forward(obs_batch)
        new_log_prob = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_prob - log_prob_batch)
        surr1 = ratio * advantage_batch
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_batch
        pi_loss = -torch.min(surr1, surr2).mean()
        value_loss = self.value_coeff * F.mse_loss(value, return_batch)
        entropy_loss = -self.entropy_coeff * entropy
        total_loss = pi_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        approx_kl = ((ratio - 1) - (new_log_prob - log_prob_batch)).mean()
        return pi_loss, value_loss, total_loss, approx_kl, torch.exp(self.log_std)

###########################################
# Buffer Implementation
###########################################

def compute_gae(rewards: np.ndarray, values: np.ndarray, last_value: float, is_last_terminal: bool, gamma: float, gae_lambda: float):
    rewards = np.array(rewards, dtype=np.float32)[:, 0]  # [N, 1] -> [N]
    values = np.array(values, dtype=np.float32)[:, 0]    # [N, 1] -> [N]
    N = len(rewards)

    # Initialize arrays
    deltas = np.zeros(N, dtype=np.float32)
    advantages = np.zeros(N, dtype=np.float32)

    # Compute deltas
    next_values = np.concatenate([values[1:], [last_value]])  # [N]
    next_non_terminal = np.ones(N, dtype=np.float32)
    if is_last_terminal:
        next_non_terminal[-1] = 0
    deltas = rewards + gamma * next_values * next_non_terminal - values

    # Vectorized GAE computation (loop for sequential dependency)
    advantages[-1] = deltas[-1]
    for t in reversed(range(N-1)):
        advantages[t] = deltas[t] + gamma * gae_lambda * next_non_terminal[t] * advantages[t+1]

    returns = advantages + values
    return returns[:, np.newaxis], advantages[:, np.newaxis]  # [N] -> [N, 1]

class PPOBuffer:
    def __init__(self, obs_dim, action_dim, buffer_capacity, seed=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_capacity = buffer_capacity
        self.rng = np.random.default_rng(seed=seed)

        self.obs = np.zeros((self.buffer_capacity, self.obs_dim), dtype=np.float32)
        self.action = np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32)
        self.reward = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.log_prob = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.advantage = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.values = np.zeros((self.buffer_capacity, 1), dtype=np.float32)

        self.clear()

    def clear(self):
        self.start_index = 0
        self.pointer = 0

    def record(self, obs, action, reward, values, log_prob):
        assert self.pointer < self.buffer_capacity, f"Buffer overflow: pointer={self.pointer}, capacity={self.buffer_capacity}"
        self.obs[self.pointer] = obs
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.values[self.pointer] = values
        self.log_prob[self.pointer] = log_prob
        self.pointer += 1

    def process_trajectory(self, gamma, gae_lam, is_last_terminal, last_v):
        path_slice = slice(self.start_index, self.pointer)
        values_t = self.values[path_slice]
        self.returns[path_slice], self.advantage[path_slice] = compute_gae(
            self.reward[path_slice], values_t, last_v, is_last_terminal, gamma, gae_lam
        )
        self.start_index = self.pointer

    def get_data(self):
        whole_slice = slice(0, self.pointer)
        return {
            'obs': self.obs[whole_slice],
            'action': self.action[whole_slice],
            'reward': self.reward[whole_slice],
            'values': self.values[whole_slice],
            'log_prob': self.log_prob[whole_slice],
            'return': self.returns[whole_slice],
            'advantage': self.advantage[whole_slice],
        }

    def load_data(self, data, pointer, start_index):
        self.clear()
        self.pointer = min(pointer, self.buffer_capacity)
        self.start_index = min(start_index, self.pointer)
        for key in ['obs', 'action', 'reward', 'values', 'log_prob', 'return', 'advantage']:
            if key in data and len(data[key]) > 0:
                self.__dict__[key][:self.pointer] = data[key][:self.pointer]

    def get_mini_batch(self, batch_size):
        assert batch_size <= self.pointer, "Batch size must be smaller than number of data."
        indices = np.arange(self.pointer)
        self.rng.shuffle(indices)
        split_indices = [i for i in range(batch_size, self.pointer, batch_size)]
        temp_data = {
            'obs': np.split(self.obs[indices], split_indices),
            'action': np.split(self.action[indices], split_indices),
            'reward': np.split(self.reward[indices], split_indices),
            'values': np.split(self.values[indices], split_indices),
            'log_prob': np.split(self.log_prob[indices], split_indices),
            'return': np.split(self.returns[indices], split_indices),
            'advantage': np.split(self.advantage[indices], split_indices),
        }
        return [
            {
                'obs': temp_data['obs'][k],
                'action': temp_data['action'][k],
                'reward': temp_data['reward'][k],
                'values': temp_data['values'][k],
                'log_prob': temp_data['log_prob'][k],
                'return': temp_data['return'][k],
                'advantage': temp_data['advantage'][k],
            } for k in range(len(temp_data['obs']))
        ]

###########################################
# Utility Functions
###########################################

class RunningStat:
    def __init__(self, shape, mean=None, std=None, count=0):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32) if mean is None else mean
        self.std = np.ones(shape, dtype=np.float32) if std is None else std
        self.count = count

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.std = np.sqrt(self.std**2 + (delta * delta2 - self.std**2) / self.count)

    def normalize(self, x):
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

def save_checkpoint(policy, buffer, running_stat, episode_count, mean_rewards, log_dir, checkpoint_path):
    checkpoint = {
        'actor_state_dict': policy.actor.state_dict(),
        'critic_state_dict': policy.critic.state_dict(),
        'log_std': policy.log_std,
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'buffer': buffer.get_data(),
        'buffer_pointer': buffer.pointer,
        'buffer_start_index': buffer.start_index,
        'buffer_rng': buffer.rng.__getstate__(),
        'episode_count': episode_count,
        'mean_rewards': mean_rewards,
        'random_state': save_random_state(),
        'running_stat': {
            'mean': running_stat.mean,
            'std': running_stat.std,
            'count': running_stat.count
        },
        'log_dir': log_dir
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(policy, buffer, running_stat, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, None, None
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if policy is not None:
        policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        policy.log_std = checkpoint['log_std']
        policy.optimizer = optim.Adam(policy.parameters(), lr=2.5e-4)
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if buffer is not None:
        buffer.load_data(checkpoint['buffer'], checkpoint['buffer_pointer'], checkpoint['buffer_start_index'])
        buffer.rng.__setstate__(checkpoint['buffer_rng'])
    if running_stat is not None:
        running_stat.mean = checkpoint['running_stat']['mean']
        running_stat.std = checkpoint['running_stat']['std']
        running_stat.count = checkpoint['running_stat']['count']

    if 'random_state' in checkpoint:
        load_random_state(checkpoint['random_state'])

    log_dir = checkpoint.get('log_dir', 'Log')
    return (checkpoint['episode_count'], checkpoint['mean_rewards'], log_dir)

###########################################
# Training Function
###########################################

def learn(policy: PPOPolicy, buffer: PPOBuffer, num_epochs: int, batch_size: int, writer: SummaryWriter, episode_count: int):
    for _ in range(num_epochs):
        batch_data = buffer.get_mini_batch(batch_size)
        for data in batch_data:
            obs_batch = torch.tensor(data['obs'], dtype=torch.float32)
            action_batch = torch.tensor(data['action'], dtype=torch.float32)
            log_prob_batch = torch.tensor(data['log_prob'], dtype=torch.float32)
            advantage_batch = torch.tensor(data['advantage'], dtype=torch.float32)
            return_batch = torch.tensor(data['return'], dtype=torch.float32)

            # Normalize advantages
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

            # Update
            pi_loss, v_loss, total_loss, approx_kl, std = policy.update(
                obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch
            )

            # Log metrics
            writer.add_scalar("train/pi_loss", pi_loss.item(), episode_count)
            writer.add_scalar("train/v_loss", v_loss.item(), episode_count)
            writer.add_scalar("train/total_loss", total_loss.item(), episode_count)
            writer.add_scalar("train/approx_kl", approx_kl.item(), episode_count)
            writer.add_scalar("train/std", std.mean().item(), episode_count)

    return approx_kl.item(), std.mean().item()

def train_ppo(env_name="Pendulum-v1",
              num_episodes_per_update=10,
              max_buffer_steps=10000,
              batch_size=64,
              total_episodes=5000,
              gamma=0.99,
              gae_lam=0.95,
              num_epochs=10,
              learning_rate=2.5e-4,
              clip_range=0.2,
              value_coeff=0.5,
              entropy_coeff=0.01,
              max_grad_norm=0.5,
              initial_std=1.0,
              save_interval=100,
              checkpoint_path="checkpoint.pt"):

    # Initialize environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    # Initialize TensorBoard
    episode_count, _, log_dir = load_checkpoint(None, None, None, checkpoint_path)
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("Log", f"run_{timestamp}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    print(f"Logging to TensorBoard at {log_dir}")

    # Initialize buffer and policy
    buffer = PPOBuffer(state_dim, action_dim, max_buffer_steps)
    policy = PPOPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        learning_rate=learning_rate,
        clip_range=clip_range,
        value_coeff=value_coeff,
        entropy_coeff=entropy_coeff,
        initial_std=initial_std,
        max_grad_norm=max_grad_norm
    )

    # State normalization
    running_stat = RunningStat((state_dim,))

    # Training data

    episode_rewards = []
    mean_rewards = []

    # Load checkpoint (if exists)
    if episode_count is not None:
        episode_count, mean_rewards, _ = load_checkpoint(policy, buffer, running_stat, checkpoint_path)
        print(f"Resumed training from checkpoint at episode={episode_count}")
    else:
        episode_count = 0

    while episode_count < total_episodes:

        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Normalize state
            running_stat.update(state)
            state_normalized = running_stat.normalize(state)
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(policy.device)

            # Collect data
            action, log_prob, value = policy.get_action(state_tensor)
            clipped_action = np.clip(action, lower_bound, upper_bound)
            next_state, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated
            episode_reward += reward

            # Store data
            buffer.record(state_normalized, action, reward, value, log_prob)

            state = next_state

        episode_count += 1
        episode_rewards.append(episode_reward)

        # Process trajectory for the episode
        next_state_normalized = running_stat.normalize(state)
        last_value = policy.get_values(torch.FloatTensor(next_state_normalized).unsqueeze(0).to(policy.device))
        buffer.process_trajectory(
            gamma=gamma,
            gae_lam=gae_lam,
            is_last_terminal=done,
            last_v=last_value
        )

        # Update policy after collecting enough episodes
        if episode_count % num_episodes_per_update == 0:
            approx_kl, std = learn(policy, buffer, num_epochs, batch_size, writer, episode_count)

            # Log returns
            mean_reward = np.mean(episode_rewards[-num_episodes_per_update:])
            mean_rewards.append(mean_reward)
            writer.add_scalar("misc/ep_reward_mean", mean_reward, episode_count)
            print(f"Episode {episode_count} | Avg Reward {mean_reward:.2f} | KL {approx_kl:.4f} | STD {std:.4f}")

            # Clear buffer and reset episode collection
            buffer.clear()

        # Save checkpoint
        if episode_count % save_interval == 0:
            save_checkpoint(policy, buffer, running_stat, episode_count,
                            mean_rewards, log_dir, checkpoint_path)

    # Plot reward curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.xlabel("Update")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True)
    plt.savefig("episode_reward.png")

    # Save final model
    save_checkpoint(policy, buffer, running_stat, episode_count,
                    mean_rewards, log_dir, checkpoint_path)

    writer.close()
    env.close()

###########################################
# Main Function
###########################################

if __name__ == "__main__":
    checkpoint_path="checkpoint.pt"
    train_ppo(total_episodes=4000, learning_rate=2.5e-4, checkpoint_path=checkpoint_path) # 運氣好2000 episode即可
    train_ppo(total_episodes=5000, learning_rate=1e-4, checkpoint_path=checkpoint_path)   # 運氣好2500
    train_ppo(total_episodes=6000, learning_rate=2e-5, checkpoint_path=checkpoint_path)   # 運氣好3000
