import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from Policy import PPOPolicy
from PPOBuffer import PPOBuffer
import os

# 超參數
NUM_STEPS = 2048  # 每次更新前的採集步數
BATCH_SIZE = 64   # 小批量大小
TOTAL_TIMESTEPS = NUM_STEPS * 500  # 總訓練步數
GAMMA = 0.99      # 折扣因子
GAE_LAM = 0.95    # GAE 參數
NUM_EPOCHS = 10   # 每次更新的迭代次數
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
MAX_GRAD_NORM = 0.5
INITIAL_STD = 1.0

# 狀態正則化
class RunningStat:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.count = 0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.std = np.sqrt(self.std**2 + (delta * delta2 - self.std**2) / self.count)

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

def main():
    # 初始化環境
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    # 初始化 TensorBoard
    if not os.path.exists("Log"):
        os.makedirs("Log")
    writer = SummaryWriter("Log")

    # 初始化緩衝區和策略
    buffer = PPOBuffer(state_dim, action_dim, NUM_STEPS)
    policy = PPOPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE,
        value_coeff=VALUE_COEFF,
        entropy_coeff=ENTROPY_COEFF,
        initial_std=INITIAL_STD,
        max_grad_norm=MAX_GRAD_NORM
    )

    # 狀態正則化
    running_stat = RunningStat((state_dim,))

    # 訓練數據
    state, _ = env.reset()
    episode_rewards = []
    episode_reward = 0
    episode_count = 0
    season_count = 0
    mean_rewards = []

    for t in range(TOTAL_TIMESTEPS):
        # 正則化狀態
        running_stat.update(state)
        state_normalized = running_stat.normalize(state)
        state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(policy.device)

        # 採集數據
        action, log_prob, value = policy.get_action(state_tensor)
        clipped_action = np.clip(action, lower_bound, upper_bound)
        next_state, reward, terminated, truncated, _ = env.step(clipped_action)
        done = terminated or truncated
        episode_reward += reward

        # 存儲數據
        buffer.record(state_normalized, action, reward, value, log_prob)

        state = next_state

        # 處理軌跡
        if done or (t + 1) % NUM_STEPS == 0:
            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state, _ = env.reset()

            # 計算 GAE 和回報
            next_state_normalized = running_stat.normalize(state)
            last_value = policy.get_values(torch.FloatTensor(next_state_normalized).unsqueeze(0).to(policy.device))
            buffer.process_trajectory(
                gamma=GAMMA,
                gae_lam=GAE_LAM,
                is_last_terminal=done,
                last_v=last_value
            )

        # 更新策略
        if (t + 1) % NUM_STEPS == 0:
            season_count += 1
            for epoch in range(NUM_EPOCHS):
                batch_data = buffer.get_mini_batch(BATCH_SIZE)
                for data in batch_data:
                    obs_batch = torch.tensor(data['obs'], dtype=torch.float32)
                    action_batch = torch.tensor(data['action'], dtype=torch.float32)
                    log_prob_batch = torch.tensor(data['log_prob'], dtype=torch.float32)
                    advantage_batch = torch.tensor(data['advantage'], dtype=torch.float32)
                    return_batch = torch.tensor(data['return'], dtype=torch.float32)

                    # 正則化優勢
                    advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

                    # 更新
                    pi_loss, v_loss, total_loss, approx_kl, std = policy.update(
                        obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch
                    )

                    # 記錄日誌
                    writer.add_scalar("train/pi_loss", pi_loss.item(), t)
                    writer.add_scalar("train/v_loss", v_loss.item(), t)
                    writer.add_scalar("train/total_loss", total_loss.item(), t)
                    writer.add_scalar("train/approx_kl", approx_kl.item(), t)
                    writer.add_scalar("train/std", std.mean().item(), t)

            # 清空緩衝區
            buffer.clear()

            # 記錄回報
            if episode_count > 0:
                mean_reward = np.mean(episode_rewards[-episode_count:])
                mean_rewards.append(mean_reward)
                writer.add_scalar("misc/ep_reward_mean", mean_reward, t)
                print(f"Season {season_count}: Mean Reward = {mean_reward:.2f}, KL = {approx_kl.item():.4f}, Std = {std.mean().item():.4f}")
                episode_count = 0

    # 保存模型
    torch.save(policy.model.actor.state_dict(), "ppo_actor.pth")
    torch.save(policy.model.critic.state_dict(), "ppo_critic.pth")

    # 繪製回報曲線
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.xlabel("Season")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True)
    plt.savefig("season_reward.png")

    # 保存正則化參數
    np.save("running_stat_mean.npy", running_stat.mean)
    np.save("running_stat_std.npy", running_stat.std)
    np.save("running_stat_count.npy", running_stat.count)

    writer.close()
    env.close()

if __name__ == "__main__":
    main()
