import gymnasium as gym
import numpy as np
from sac_imp import SAC
from collections import deque
import json

class SACTrainer:
    """Handles the training process for SAC algorithm"""
    def __init__(
        self,
        env_name='BipedalWalker-v3',
        max_episodes=1000,
        max_steps=1000,
        batch_size=256,
        eval_interval=50,
        updates_per_step=1,
        start_steps=10000,
        eval_episodes=50,
        debug_config = None
    ):
        # Default debugging configuration with more focused outputs
        self.debug_config = {
            'print_episode_summary': True,     # Summary of each episode
            'print_eval_summary': True,        # Summary of evaluation runs
            'print_periodic_stats': True       # Periodic training statistics
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
            lr=3e-4,
            alpha=0.2,
            automatic_entropy_tuning=True
        )
        
        # Initialize logging
        self.rewards_history = []
        self.eval_rewards_history = []
        self.episode_length_history = []
        self.loss_history = []
        
        
    
    def print_episode_summary(self, episode, total_steps, episode_reward, episode_length, rolling_reward):
        if not self.debug_config['print_episode_summary']:
            return

        # Calculate mean losses for the episode
        mean_losses = {
            'q1': np.mean(self.episode_stats['q1_losses']),
            'q2': np.mean(self.episode_stats['q2_losses']),
            'policy': np.mean(self.episode_stats['policy_losses'])
        }
        
        # Calculate action statistics
        action_means = np.mean(self.episode_stats['action_means'], axis=0)
        action_stds = np.mean(self.episode_stats['action_stds'], axis=0)

        print(f"\n{'='*50}")
        print(f"Episode {episode} Summary:")
        print(f"Steps: {total_steps} | Length: {episode_length}")
        print(f"Reward: {episode_reward:.2f} | Rolling Avg: {rolling_reward:.2f}")
        print(f"Mean Losses - Q1: {mean_losses['q1']:.4f}, Q2: {mean_losses['q2']:.4f}, Policy: {mean_losses['policy']:.4f}")
        print(f"Action Means: {action_means}")
        print(f"Action StDevs: {action_stds}")
        print(f"{'='*50}")

        # Clear episode statistics for next episode
        for key in self.episode_stats:
            self.episode_stats[key] = []

    def update_episode_stats(self, losses, actions):
        """Updates episode statistics during training"""
        if losses:
            self.episode_stats['q1_losses'].append(losses['q1_loss'])
            self.episode_stats['q2_losses'].append(losses['q2_loss'])
            self.episode_stats['policy_losses'].append(losses['policy_loss'])
        
        if actions is not None:
            self.episode_stats['action_means'].append(np.mean(actions, axis=0))
            self.episode_stats['action_stds'].append(np.std(actions, axis=0))

    def debug_print(self, category, message):
        if self.debug_config.get(f'print_{category}', False):
            print(f"[DEBUG-{category}] {message}")

    def evaluate_policy(self):
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

        if self.debug_config['print_eval_summary']:
            print(f"\n{'='*50}")
            print(f"Evaluation Summary:")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Mean Episode Length: {np.mean(eval_lengths):.1f}")
            print(f"Success Rate: {sum(r > 300 for r in eval_rewards)/len(eval_rewards):.2%}")
            print(f"{'='*50}")

        return mean_reward, std_reward

    def train(self, start_episode=0, total_steps=0):
        """
        Main training loop for SAC algorithm.
        
        Args:
            start_episode (int): Episode number to start/resume from
            total_steps (int): Total number of steps taken in previous training
        """
        best_eval_reward = getattr(self, 'best_eval_reward', float('-inf'))
        early_stop_patience = 1000
        no_improvement_count = 0
        rolling_reward = deque(maxlen=100)
        
        print(f"\nStarting training from episode {start_episode}")
        print(f"State dim: {self.env.observation_space.shape[0]}")
        print(f"Action dim: {self.env.action_space.shape[0]}")
        print(f"Training parameters:")
        print(f"  Max episodes: {self.max_episodes}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Updates per step: {self.updates_per_step}")
        print("\n" + "="*50)
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
                        self.loss_history.append(update_info)
                
                if episode_steps >= self.max_steps:
                    done = True
            
            # Store episode information
            self.rewards_history.append(episode_reward)
            self.episode_length_history.append(episode_steps)
            rolling_reward.append(episode_reward)
            
            # Print episode information #TODO: add more info
            if self.debug_config['print_episode_summary']:
                print(f"\nEpisode {episode} - Steps: {total_steps}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Average Reward: {np.mean(rolling_reward):.2f}")
                print(f"Episode steps: {episode_steps}")
                print(f"Action mean: {np.mean(action):.2f}")
                print(f"Action std: {np.std(action):.2f}")
                #print(f"Q1 loss: {update_info['q1_loss']:.4f}")
                #print(f"Q2 loss: {update_info['q2_loss']:.4f}")
                #print(f"Policy loss: {update_info['policy_loss']:.4f}")
                
            # Evaluate policy
            if episode % self.eval_interval == 0 and episode>2:
                eval_reward, eval_std = self.evaluate_policy()
                self.eval_rewards_history.append(eval_reward)
                
                # Save if best performance
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    if hasattr(self, 'save_best_model'):
                        self.save_best_model()
                    no_improvement_count = 0
                else: 
                    no_improvement_count += 1
                
                # Save checkpoint if available
                if hasattr(self, 'save_checkpoint'):
                    self.save_checkpoint(episode, total_steps)
            
            # Early stopping check
            if no_improvement_count >= early_stop_patience:
                print("\nNo improvement for a while. Stopping training.")
                break
        
        print("\nTraining completed!")
        print(f"Total steps: {total_steps}")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        print(f"Final average reward: {np.mean(rolling_reward):.2f}")
        
    def save_training_history(self):
        """Saves training metrics to a JSON file"""
        history = {
            'rewards': self.rewards_history,
            'eval_rewards': self.eval_rewards_history,
            'episode_lengths': self.episode_length_history,
            'losses': self.loss_history
        }
        
        with open(f"{self.save_dir}/training_history.json", 'w') as f:
            json.dump(history, f)

