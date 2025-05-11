from collections import deque
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac import SAC
from dmc import make_dmc_env

def save_best_model(agent: SAC, save_dir: str):
    """Save the best model based on evaluation reward"""
    save_path = os.path.join(save_dir, 'best_model.pth')
    torch.save(agent.policy.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

def save_checkpoint(agent: SAC, episode: int, total_steps: int, log_dir: str, save_dir: str):
    """Save a checkpoint of the current training state"""
    checkpoint_path = os.path.join(save_dir, f'episode_{episode}.pth')
    checkpoint = {
        'agent_state_dict'  : agent.state_dict(),
        'episode'           : episode,
        'total_steps'       : total_steps,
        'log_dir'           : log_dir
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(agent: SAC, checkpoint_path: str):
    """Load a checkpoint to resume training"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} does not exist")
        return None, None, None

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    start_episode = checkpoint.get('episode', 0)
    total_steps = checkpoint.get('total_steps', 0)
    log_dir = checkpoint['log_dir']
    print(f"Resume from Episode {start_episode}, Total Steps {total_steps}")
    return start_episode, total_steps, log_dir

def train(
    env,
    agent: SAC,
    start_episode: int,
    max_episodes: int,
    total_steps: int,
    batch_size: int,
    log_dir: str,
    save_dir: str,
    save_interval,
):
    print(f"Start Training from Steps: {total_steps}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    rolling_rewards = deque(maxlen=100)

    for episode in range(start_episode + 1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward
            total_steps += 1

            if agent.replay_buffer.size > batch_size:
                policy_loss, q1_loss, q2_loss, forward_loss, inverse_loss, intrinsic_reward = agent.learn(batch_size)

                writer.add_scalar('Losses/Policy', policy_loss, total_steps)
                writer.add_scalar('Losses/Q1', q1_loss, total_steps)
                writer.add_scalar('Losses/Q2', q2_loss, total_steps)
                writer.add_scalar('Losses/Forward', forward_loss, total_steps)
                writer.add_scalar('Losses/Inverse', inverse_loss, total_steps)
                writer.add_scalar('Rewards/Intrinsic', intrinsic_reward, total_steps)
                writer.add_scalar('Misc/Alpha', agent.alpha, total_steps)
                # writer.add_scalar('Action/Mean', np.mean(action), total_steps)
                # writer.add_scalar('Action/Std', np.std(action), total_steps)

        rolling_rewards.append(episode_reward)
        writer.add_scalar('Rewards/Episode', episode_reward, episode)
        writer.add_scalar('Rewards/Avg100Episodes', np.mean(rolling_rewards), episode)

        # Save checkpoint
        if episode % save_interval == 0:
            save_checkpoint(agent, episode, total_steps, log_dir, save_dir)

    writer.close()

if __name__ == "__main__":
    import random
    seed = np.random.randint(0, 10000)
    print(f"Seed = {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_dmc_env('humanoid-walk', seed, flatten=True, use_pixels=False)

    agent = SAC(
        state_dim       = env.observation_space.shape[0],
        action_dim      = env.action_space.shape[0],
        hidden_dim      = 256,
        action_bounds   = (float(env.action_space.low[0]), float(env.action_space.high[0])),
        gamma           = 0.99,
        tau             = 0.005,
        lr              = 1e-4,
        alpha           = 0.2,
        icm_eta         = 0.1,
        forward_weight  = 1.0,
        inverse_weight  = 0.1,
        buffer_capacity = 1_000_000,
    )

    save_dir        = "./checkpoints"
    checkpoint_path = "./checkpoints/episode_{}.pth"
    start_episode, total_steps, log_dir = load_checkpoint(agent, checkpoint_path)
    if start_episode is None:
        start_episode = 0
    if total_steps is None:
        total_steps = 0
    if log_dir is None:
        log_dir = f"Logs/run_{int(time.time())}"

    train(
        env             = env,
        agent           = agent,
        start_episode   = start_episode,
        max_episodes    = 10_000,
        total_steps     = total_steps,
        batch_size      = 128,
        log_dir         = log_dir,
        save_dir        = save_dir,
        save_interval   = 50,
    )
