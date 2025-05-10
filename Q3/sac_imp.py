import torch
import torch.nn.functional as F
import torch.optim as optim
from networks_model1 import Critic, Actor
from replay_buffer import ReplayBuffer

class SAC:
    """Soft Actor-Critic implementation for continuous action spaces"""
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,  # 降低學習率以適配複雜環境
        alpha=0.2,
        automatic_entropy_tuning=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize the networks
        self.policy = Actor(state_dim, action_dim, hidden_dim, action_bounds=(-1.0, 1.0)).to(device)
        self.q1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = Critic(state_dim, action_dim, hidden_dim).to(device)

        # Create target networks
        self.q1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim  # Target entropy = -|A|
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Initialize replay buffer for humanoid-walk
        self.replay_buffer = ReplayBuffer(capacity=1000000, state_dim=state_dim, action_dim=action_dim)

    def select_action(self, state, evaluate=False):
        """Select an action from the input state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.policy(state)
                action = torch.tanh(mean)
                action = action * self.policy.action_scale + self.policy.action_bias
                return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _ = self.policy.sample(state)
                return action.cpu().numpy()[0]

    def update_parameters(self, batch_size=256):
        """Update the networks using a batch of experiences"""
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state_batch)
            q1_next = self.q1_target(next_state_batch, next_actions)
            q2_next = self.q2_target(next_state_batch, next_actions)
            q_next = torch.min(q1_next, q2_next)
            value_target = q_next - self.alpha * next_log_probs
            q_target = reward_batch + (1 - done_batch) * self.gamma * value_target

        q1_pred = self.q1(state_batch, action_batch)
        q2_pred = self.q2(state_batch, action_batch)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.policy.sample(state_batch)
        q1_new = self.q1(state_batch, new_actions)
        q2_new = self.q2(state_batch, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        self._soft_update_target_networks()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item()
        }

    def _soft_update_target_networks(self):
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'alpha': self.alpha
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.alpha = checkpoint['alpha']

    def save_checkpoint(self, path, episode, total_steps, replay_buffer=True):
        checkpoint = {
            'episode': episode,
            'total_steps': total_steps,
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'alpha': self.alpha
        }
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        if replay_buffer:
            checkpoint['replay_buffer'] = (
                self.replay_buffer.states[:self.replay_buffer.size],
                self.replay_buffer.actions[:self.replay_buffer.size],
                self.replay_buffer.rewards[:self.replay_buffer.size],
                self.replay_buffer.next_states[:self.replay_buffer.size],
                self.replay_buffer.dones[:self.replay_buffer.size]
            )
            checkpoint['replay_buffer_size'] = self.replay_buffer.size
            checkpoint['replay_buffer_pos'] = self.replay_buffer.pos
            checkpoint['replay_buffer_rng'] = self.replay_buffer.rng.__getstate__()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path, load_replay_buffer=True):
        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        if 'policy_optimizer_state_dict' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if 'q1_optimizer_state_dict' in checkpoint:
            self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        if 'q2_optimizer_state_dict' in checkpoint:
            self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            if 'alpha_optimizer_state_dict' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        if load_replay_buffer and 'replay_buffer' in checkpoint:
            states, actions, rewards, next_states, dones = checkpoint['replay_buffer']
            self.replay_buffer.states[:len(states)] = states
            self.replay_buffer.actions[:len(actions)] = actions
            self.replay_buffer.rewards[:len(rewards)] = rewards
            self.replay_buffer.next_states[:len(next_states)] = next_states
            self.replay_buffer.dones[:len(dones)] = dones
            self.replay_buffer.size = checkpoint['replay_buffer_size']
            self.replay_buffer.pos = checkpoint['replay_buffer_pos']
            if 'replay_buffer_rng' in checkpoint:
                self.replay_buffer.rng.__setstate__(checkpoint['replay_buffer_rng'])
        return checkpoint.get('episode', 0), checkpoint.get('total_steps', 0)
