import numpy as np
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque
from sac_imp import SAC
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

class SACTrainer:
    """Handles the training process for SAC algorithm with TensorBoard logging"""
    def __init__(
        self,
        env_name='humanoid-walk',
        max_episodes=1000,  # 增加回合數以適配更複雜環境
        batch_size=256,     # 增大批量大小以穩定訓練
        eval_interval=50,   # 增加評估間隔以減少計算開銷
        updates_per_step=1,
        start_steps=10000,  # 增加隨機探索步數
        eval_episodes=5,    # 減少評估回合數以節省時間
        save_dir='checkpoints',
        log_dir=f'Logs/run_{int(time.time())}',
        checkpoint_path=None,
        debug_config=None
    ):
        # Default debugging configuration
        self.debug_config = {
            'log_tensorboard': True,
            'print_minimal': True
        }
        if debug_config is not None:
            self.debug_config.update(debug_config)

        # Storage for episode statistics
        self.episode_stats = {
            'q1_losses': [],
            'q2_losses': [],
            'policy_losses': [],
            'action_means': [],
            'action_stds': []
        }
        # Initialize training parameters
        self.env_name = env_name
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps
        self.eval_episodes = eval_episodes
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.checkpoint_path = checkpoint_path

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Create environments using DMC
        self.env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
        self.eval_env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)

        # Get environment dimensions
        state_dim = self.env.observation_space.shape[0]  # 67 for humanoid-walk
        action_dim = self.env.action_space.shape[0]      # 21 for humanoid-walk

        # Initialize SAC agent
        self.agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            gamma=0.99,
            tau=0.005,
            lr=3e-4,  # 降低學習率以適配複雜環境
            alpha=0.2,
            automatic_entropy_tuning=True
        )

        # Initialize TensorBoard writer (log_dir may be updated by checkpoint)
        self.writer = None

        # Load checkpoint if provided
        self.start_episode = 0
        self.total_steps = 0
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            if self.debug_config['print_minimal']:
                print(f"Resumed training from checkpoint: {checkpoint_path}")
                print(f"Starting from episode {self.start_episode}, total steps {self.total_steps}")

        # Initialize TensorBoard writer if not set by checkpoint
        if self.writer is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)

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

    def save_checkpoint(self, episode, total_steps):
        """Save a checkpoint of the current training state"""
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pth')
        checkpoint = {
            'episode': episode,
            'total_steps': total_steps,
            'policy_state_dict': self.agent.policy.state_dict(),
            'q1_state_dict': self.agent.q1.state_dict(),
            'q2_state_dict': self.agent.q2.state_dict(),
            'q1_target_state_dict': self.agent.q1_target.state_dict(),
            'q2_target_state_dict': self.agent.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.agent.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.agent.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.agent.q2_optimizer.state_dict(),
            'alpha': self.agent.alpha,
            'log_alpha': self.agent.log_alpha if self.agent.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.agent.alpha_optimizer.state_dict() if self.agent.automatic_entropy_tuning else None,
            'replay_buffer': (
                self.agent.replay_buffer.states[:self.agent.replay_buffer.size],
                self.agent.replay_buffer.actions[:self.agent.replay_buffer.size],
                self.agent.replay_buffer.rewards[:self.agent.replay_buffer.size],
                self.agent.replay_buffer.next_states[:self.agent.replay_buffer.size],
                self.agent.replay_buffer.dones[:self.agent.replay_buffer.size]
            ),
            'replay_buffer_size': self.agent.replay_buffer.size,
            'replay_buffer_pos': self.agent.replay_buffer.pos,
            'replay_buffer_rng': self.agent.replay_buffer.rng.__getstate__(),
            'random_state': self.save_random_state(),
            'log_dir': self.log_dir
        }
        torch.save(checkpoint, checkpoint_path)
        if self.debug_config['print_minimal']:
            print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Load agent state
        self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.agent.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.agent.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.agent.alpha = checkpoint['alpha']
        if self.agent.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.agent.log_alpha = checkpoint['log_alpha']
            self.agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

        # Load replay buffer
        if 'replay_buffer' in checkpoint:
            states, actions, rewards, next_states, dones = checkpoint['replay_buffer']
            self.agent.replay_buffer.states[:len(states)] = states
            self.agent.replay_buffer.actions[:len(actions)] = actions
            self.agent.replay_buffer.rewards[:len(rewards)] = rewards
            self.agent.replay_buffer.next_states[:len(next_states)] = next_states
            self.agent.replay_buffer.dones[:len(dones)] = dones
            self.agent.replay_buffer.size = checkpoint['replay_buffer_size']
            self.agent.replay_buffer.pos = checkpoint['replay_buffer_pos']
            self.agent.replay_buffer.rng.__setstate__(checkpoint['replay_buffer_rng'])

        # Load random state
        if 'random_state' in checkpoint:
            self.load_random_state(checkpoint['random_state'])

        # Load training progress
        self.start_episode = checkpoint.get('episode', 0)
        self.total_steps = checkpoint.get('total_steps', 0)

        # Load log directory for TensorBoard continuity
        self.log_dir = checkpoint.get('log_dir', self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_episode_summary(self, episode, total_steps, episode_reward, episode_length, rolling_reward):
        """Log episode statistics to TensorBoard"""
        if not self.debug_config['log_tensorboard']:
            return

        # Calculate mean losses for the episode
        mean_losses = {
            'q1': np.mean(self.episode_stats['q1_losses']) if self.episode_stats['q1_losses'] else 0.0,
            'q2': np.mean(self.episode_stats['q2_losses']) if self.episode_stats['q2_losses'] else 0.0,
            'policy': np.mean(self.episode_stats['policy_losses']) if self.episode_stats['policy_losses'] else 0.0
        }

        # Calculate action statistics
        action_mean = np.mean(self.episode_stats['action_means']) if self.episode_stats['action_means'] else 0.0
        action_std = np.mean(self.episode_stats['action_stds']) if self.episode_stats['action_stds'] else 0.0

        # Log to TensorBoard
        self.writer.add_scalar('Rewards/Episode', episode_reward, episode)
        self.writer.add_scalar('Rewards/Rolling_Avg', rolling_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        self.writer.add_scalar('Losses/Q1', mean_losses['q1'], episode)
        self.writer.add_scalar('Losses/Q2', mean_losses['q2'], episode)
        self.writer.add_scalar('Losses/Policy', mean_losses['policy'], episode)
        self.writer.add_scalar('Actions/Mean', action_mean, episode)
        self.writer.add_scalar('Actions/Std', action_std, episode)
        self.writer.add_scalar('Hyperparams/Alpha', self.agent.alpha, episode)

        # Clear episode statistics for next episode
        for key in self.episode_stats:
            self.episode_stats[key] = []

    def update_episode_stats(self, losses, action):
        """Updates episode statistics during training"""
        if losses:
            self.episode_stats['q1_losses'].append(losses['q1_loss'])
            self.episode_stats['q2_losses'].append(losses['q2_loss'])
            self.episode_stats['policy_losses'].append(losses['policy_loss'])

        if action is not None:
            self.episode_stats['action_means'].append(np.mean(action))
            self.episode_stats['action_stds'].append(np.std(action))

    def evaluate_policy(self, episode):
        """Evaluate the policy and log results to TensorBoard"""
        eval_rewards = []
        eval_lengths = []

        for eval_ep in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                state = next_state

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)

        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)

        if self.debug_config['log_tensorboard']:
            self.writer.add_scalar('Rewards/Eval_Mean', mean_reward, episode)
            self.writer.add_scalar('Rewards/Eval_Std', std_reward, episode)
            self.writer.add_scalar('Episode/Eval_Length', np.mean(eval_lengths), episode)

        return mean_reward, std_reward

    def save_best_model(self):
        """Save the best model based on evaluation reward"""
        save_path = os.path.join(self.save_dir, 'best_model.pth')
        self.agent.save(save_path)
        if self.debug_config['print_minimal']:
            print(f"Saved best model to {save_path}")

    def train(self, start_episode=None, total_steps=None):
        """
        Main training loop for SAC algorithm.

        Args:
            start_episode (int): Episode number to start/resume from
            total_steps (int): Total number of steps taken in previous training
        """
        # Use loaded checkpoint values if not provided
        if start_episode is None:
            start_episode = self.start_episode
        if total_steps is None:
            total_steps = self.total_steps

        best_eval_reward = float('-inf')
        early_stop_patience = 100  # 增加耐心以適配複雜環境
        no_improvement_count = 0
        rolling_reward = deque(maxlen=100)

        if self.debug_config['print_minimal']:
            print(f"\nStarting training for {self.env_name} from episode {start_episode}")
            print(f"State dim: {self.env.observation_space.shape[0]}, Action dim: {self.env.action_space.shape[0]}")
            print(f"Save directory: {self.save_dir}, Log directory: {self.log_dir}")
            print(f"Total steps so far: {total_steps}")

        for episode in range(start_episode + 1, self.max_episodes + 1):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                # Select action: random for exploration or from policy
                if total_steps < self.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.agent.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Update networks
                if len(self.agent.replay_buffer) > self.batch_size:
                    for _ in range(self.updates_per_step):
                        update_info = self.agent.update_parameters(self.batch_size)
                        self.update_episode_stats(update_info, action)

            # Log episode information to TensorBoard
            rolling_reward.append(episode_reward)
            self.log_episode_summary(
                episode,
                total_steps,
                episode_reward,
                episode_steps,
                np.mean(rolling_reward) if rolling_reward else episode_reward
            )

            # Evaluate policy and save checkpoint
            if episode % self.eval_interval == 0 and episode > 2:
                eval_reward, eval_std = self.evaluate_policy(episode)

                # Save if best performance
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_best_model()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Save checkpoint
                self.save_checkpoint(episode, total_steps)

            # Save checkpoint at the last episode
            if episode == self.max_episodes - 1:
                self.save_checkpoint(episode, total_steps)

            # Early stopping check
            if no_improvement_count >= early_stop_patience:
                if self.debug_config['print_minimal']:
                    print("\nNo improvement for a while. Stopping training.")
                break

        if self.debug_config['print_minimal']:
            print("\nTraining completed!")
            print(f"Total steps: {total_steps}")
            print(f"Best evaluation reward: {best_eval_reward:.2f}")
            print(f"Final average reward: {np.mean(rolling_reward):.2f}")

        # Close TensorBoard writer
        self.writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SAC on humanoid-walk with TensorBoard logging")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()

    # Initialize trainer with optional checkpoint
    trainer = SACTrainer(checkpoint_path=args.checkpoint)
    # Start training
    trainer.train()
