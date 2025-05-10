import torch
import torch.nn.functional as F
import torch.optim as optim
from actor_critic import Critic, Actor
from replay_buffer import ReplayBuffer
from typing import Tuple, Dict, Any

class SAC:
    """Soft Actor-Critic for continuous action spaces"""
    def __init__(
        self,
        state_dim       : int,
        action_dim      : int,
        hidden_dim      : int                   = 256,
        action_bounds   : Tuple[float, float]   = (-1.0, 1.0),
        lr              : float                 = 3e-4,
        gamma           : float                 = 0.99,
        tau             : float                 = 0.005,
        alpha           : float                 = 0.2,
        buffer_capacity : int                   = 1_000_000,
        device          : torch.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        self.device = device
        self.gamma  = gamma
        self.tau    = tau
        self.alpha  = alpha

        # Networks
        self.policy = Actor (state_dim, action_dim, hidden_dim, action_bounds).to(device)
        self.q1     = Critic(state_dim, action_dim, hidden_dim)               .to(device)
        self.q2     = Critic(state_dim, action_dim, hidden_dim)               .to(device)

        # Target networks
        self.q1_target = Critic(state_dim, action_dim, hidden_dim)            .to(device)
        self.q2_target = Critic(state_dim, action_dim, hidden_dim)            .to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer   = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer       = optim.Adam(self.q1.parameters()    , lr=lr)
        self.q2_optimizer       = optim.Adam(self.q2.parameters()    , lr=lr)

        # Automatic entropy tuning
        self.target_entropy     = -action_dim  # Target entropy = -|A|
        self.log_alpha          = torch.zeros(1, requires_grad=True  , device=device)
        self.alpha_optimizer    = optim.Adam([self.log_alpha]        , lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim, action_dim=action_dim)

    def select_action(self, state: torch.Tensor, evaluate: bool=False) -> float:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate: action    = self.policy.act(state)
            else:        action, _ = self.policy.sample(state)
            return action.cpu().numpy()[0]

    def soft_update(self) -> None:
        for target_p, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)
        for target_p, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)

    def learn(self, batch_size: int) -> None:
        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.sample(batch_size)

        state_b         = torch.FloatTensor(state_b)     .to(self.device)
        action_b        = torch.FloatTensor(action_b)    .to(self.device)
        reward_b        = torch.FloatTensor(reward_b)    .to(self.device)
        next_state_b    = torch.FloatTensor(next_state_b).to(self.device)
        done_b          = torch.FloatTensor(done_b)      .to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state_b)
            q1_next         = self.q1_target(next_state_b, next_actions)
            q2_next         = self.q2_target(next_state_b, next_actions)
            q_next          = torch.min(q1_next, q2_next)
            value_target    = q_next - self.alpha * next_log_probs
            q_target        = reward_b + (1 - done_b) * self.gamma * value_target

        q1_pred = self.q1(state_b, action_b)
        q2_pred = self.q2(state_b, action_b)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.policy.sample(state_b)
        q1_new  = self.q1(state_b, new_actions)
        q2_new  = self.q2(state_b, new_actions)
        q_new   = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Entropy Tuning
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.soft_update()

        return policy_loss.item(), q1_loss.item(), q2_loss.item()

    def state_dict(self, replay_buffer: bool=True) -> Dict[str, Any]:
        state_dict = {
            'policy_state_dict'             : self.policy.state_dict(),
            'q1_state_dict'                 : self.q1.state_dict(),
            'q2_state_dict'                 : self.q2.state_dict(),
            'q1_target_state_dict'          : self.q1_target.state_dict(),
            'q2_target_state_dict'          : self.q2_target.state_dict(),
            'policy_optimizer_state_dict'   : self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict'       : self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict'       : self.q2_optimizer.state_dict(),
            'alpha'                         : self.alpha,
            'log_alpha'                     : self.log_alpha,
            'alpha_optimizer_state_dict'    : self.alpha_optimizer.state_dict(),
        }
        if replay_buffer:
            state_dict['replay_buffer'] = self.replay_buffer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], load_replay_buffer: bool=True) -> None:
        self.policy             .load_state_dict(state_dict['policy_state_dict'])
        self.q1                 .load_state_dict(state_dict['q1_state_dict'])
        self.q2                 .load_state_dict(state_dict['q2_state_dict'])
        self.q1_target          .load_state_dict(state_dict['q1_target_state_dict'])
        self.q2_target          .load_state_dict(state_dict['q2_target_state_dict'])
        self.policy_optimizer   .load_state_dict(state_dict['policy_optimizer_state_dict'])
        self.q1_optimizer       .load_state_dict(state_dict['q1_optimizer_state_dict'])
        self.q2_optimizer       .load_state_dict(state_dict['q2_optimizer_state_dict'])
        self.alpha              = state_dict['alpha']
        self.log_alpha          = state_dict['log_alpha']
        self.alpha_optimizer    .load_state_dict(state_dict['alpha_optimizer_state_dict'])
        if load_replay_buffer and 'replay_buffer' in state_dict:
            self.replay_buffer  .load_state_dict(state_dict['replay_buffer'])
