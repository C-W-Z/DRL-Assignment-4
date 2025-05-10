import gymnasium as gym
import numpy as np
from sac_imp import SAC
from collections import deque
import json
import os
from torch.utils.tensorboard import SummaryWriter

class SACTrainer:
    """Handles the training process for SAC algorithm with TensorBoard logging"""
    def __init__(
        self,
        env_name='Pendulum-v1',
        max_episodes=200,
        max_steps=200,
        batch_size=64,
        eval_interval=10,
        updates_per_step=1,
        start_steps=1000,
        eval_episodes=10,
        save_dir='checkpoints',
        log_dir='runs',
        debug_config=None
    ):
        # Default debugging configuration
        self.debug_config = {
            'log_tensorboard': True,  # Enable TensorBoard logging
            'print_minimal': True     # Minimal console output
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
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps
        self.eval_episodes = eval_episodes
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Create save and log directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        # Create environments
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)

        # Get environment dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Initialize SAC agent
        self.agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            gamma=0.99,
            tau=0.005,
            lr=1e-3,
            alpha=0.2,
            automatic_entropy_tuning=True
        )

        # Initialize logging
        self.rewards_history = []
        self.eval_rewards_history = []
        self.episode_length_history = []
        self.loss_history = []

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

    def save_checkpoint(self, episode, total_steps):
        """Save a checkpoint of the current training state"""
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pth')
        self.agent.save_checkpoint(checkpoint_path, episode, total_steps)
        if self.debug_config['print_minimal']:
            print(f"Saved checkpoint to {checkpoint_path}")

    def train(self, start_episode=0, total_steps=0):
        """
        Main training loop for SAC algorithm.

        Args:
            start_episode (int): Episode number to start/resume from
            total_steps (int): Total number of steps taken in previous training
        """
        best_eval_reward = getattr(self, 'best_eval_reward', float('-inf'))
        early_stop_patience = 50
        no_improvement_count = 0
        rolling_reward = deque(maxlen=100)

        if self.debug_config['print_minimal']:
            print(f"\nStarting training for {self.env_name} from episode {start_episode}")
            print(f"State dim: {self.env.observation_space.shape[0]}, Action dim: {self.env.action_space.shape[0]}")
            print(f"Save directory: {self.save_dir}, Log directory: {self.log_dir}")
            print(f"Total steps so far: {total_steps}")

        for episode in range(start_episode, self.max_episodes):
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
                        self.loss_history.append(update_info)

                if episode_steps >= self.max_steps:
                    done = True

            # Store episode information
            self.rewards_history.append(episode_reward)
            self.episode_length_history.append(episode_steps)
            rolling_reward.append(episode_reward)

            # Log episode information to TensorBoard
            self.log_episode_summary(
                episode,
                total_steps,
                episode_reward,
                episode_steps,
                np.mean(rolling_reward) if rolling_reward else episode_reward
            )

            # Evaluate policy
            if episode % self.eval_interval == 0 and episode > 2:
                eval_reward, eval_std = self.evaluate_policy(episode)
                self.eval_rewards_history.append(eval_reward)

                # Save if best performance
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_best_model()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Save checkpoint
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

        # Save training history
        self.save_training_history()
        # Close TensorBoard writer
        self.writer.close()

    def save_training_history(self):
        """Saves training metrics to a JSON file"""
        history = {
            'rewards': self.rewards_history,
            'eval_rewards': self.eval_rewards_history,
            'episode_lengths': self.episode_length_history,
            'losses': self.loss_history
        }

        save_path = os.path.join(self.save_dir, 'training_history.json')
        with open(save_path, 'w') as f:
            json.dump(history, f)
        if self.debug_config['print_minimal']:
            print(f"Saved training history to {save_path}")

if __name__ == "__main__":
    # Initialize trainer with default parameters
    trainer = SACTrainer()
    # Start training
    trainer.train()
