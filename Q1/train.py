import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm

# Actor-Critic 網絡（不變）
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feature = self.feature(state)
        mean = torch.tanh(self.actor_mean(feature)) * 2.0  # 縮放到 [-2, 2]
        std = self.log_std.exp().clamp(1e-6, 50.0)  # 限制標準差範圍
        value = self.critic(feature)
        return mean, std, value

# PPO 訓練類
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, gae_lambda=0.95, epochs=10, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        # 標準化優勢函數以提高訓練穩定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, states, actions, log_probs_old, advantages, returns):
        # 將輸入數據移到 device
        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        for _ in range(self.epochs):
            # 隨機打亂索引
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 重新計算當前策略的輸出（創建新的計算圖）
                mean, std, value = self.model(batch_states)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                # PPO 裁剪損失
                ratios = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(value.squeeze(), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # 清除梯度並進行反向傳播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, env: gym.Env, total_timesteps=100000, max_steps=200):
        state, _ = env.reset()
        episode_rewards = []
        t = 0

        while t < total_timesteps:
            # 初始化數據緩衝區
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
            episode_reward = 0

            for _ in range(max_steps):
                state_tensor = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():  # 數據採集時不計算梯度
                    mean, std, value = self.model(state_tensor.unsqueeze(0))
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)

                # 執行環境步驟
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
                done = terminated or truncated
                episode_reward += reward

                # 儲存數據（確保不保存計算圖）
                states.append(state)
                actions.append(action.cpu().numpy()[0])
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())
                dones.append(done)

                state = next_state
                t += 1

                if done:
                    state, _ = env.reset()
                    episode_rewards.append(episode_reward)
                    print(f"Episode {len(episode_rewards)}: Reward = {episode_reward:.2f}")
                    episode_reward = 0
                    break

            # 將數據轉為 tensor
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            log_probs = torch.FloatTensor(log_probs).to(self.device)
            rewards = rewards  # 保持為列表，供 GAE 使用
            values = values    # 保持為列表
            dones = dones      # 保持為列表

            # 計算 GAE 和 returns
            next_state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                _, _, next_value = self.model(next_state_tensor.unsqueeze(0))
            advantages, returns = self.compute_gae(rewards, values, next_value.item(), dones)

            # 更新策略
            self.update(states, actions, log_probs, advantages, returns)

        # 保存模型
        torch.save(self.model.state_dict(), "ppo_pendulum.pth")
        return episode_rewards

# 主訓練函數
def main():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo = PPO(state_dim, action_dim)
    ppo.train(env, total_timesteps=100000)
    env.close()

if __name__ == "__main__":
    main()
