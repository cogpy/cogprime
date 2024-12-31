import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque
import random

@dataclass
class Experience:
    """Represents a single learning experience"""
    state: torch.Tensor
    action: str
    reward: float
    next_state: torch.Tensor
    done: bool

class ExperienceBuffer:
    """Stores and manages learning experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class ReinforcementLearner(nn.Module):
    """Implements reinforcement learning mechanisms"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.feature_dim = self.config.get('feature_dim', 512)
        self.action_dim = self.config.get('action_dim', 128)
        
        # Action mapping
        self.action_to_idx = {
            'focus_attention': 0,
            'update_goal': 1,
            'query_memory': 2,
            'external_action': 3
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        
        # Experience replay buffer
        self.experience_buffer = ExperienceBuffer(
            capacity=self.config.get('buffer_capacity', 10000)
        )
        
        # Q-Network for action-value estimation
        self.q_network = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.action_to_idx))  # One output per action
        )
        
        # Target network for stable learning
        self.target_network = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.action_to_idx))  # One output per action
        )
        
        # Copy initial weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        
        # Learning parameters
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.epsilon = self.config.get('epsilon', 0.1)  # Exploration rate
        self.batch_size = self.config.get('batch_size', 32)
        self.update_target_every = self.config.get('update_target_every', 100)
        self.steps = 0
    
    def select_action(self, state: torch.Tensor) -> Tuple[str, float]:
        """Select action using epsilon-greedy policy with optimistic exploration"""
        if random.random() < self.epsilon:
            # Explore: weighted random action based on historical success
            action_weights = torch.ones(len(self.action_to_idx))
            if hasattr(self, '_action_success_count'):
                total_attempts = sum(self._action_success_count.values()) + len(self.action_to_idx)
                for action_idx in range(len(self.action_to_idx)):
                    action_name = self.idx_to_action[action_idx]
                    success_rate = (self._action_success_count.get(action_name, 0) + 1) / total_attempts
                    action_weights[action_idx] = success_rate
            
            # Normalize weights
            action_weights = action_weights / action_weights.sum()
            
            # Sample action using weights
            action_idx = torch.multinomial(action_weights, 1).item()
            confidence = self.epsilon * float(action_weights[action_idx])
        else:
            # Exploit: best known action with UCB-style exploration bonus
            with torch.no_grad():
                q_values = self.q_network(state)
                
                # Add exploration bonus based on action frequency
                if hasattr(self, '_action_counts'):
                    total_steps = sum(self._action_counts.values()) + len(self.action_to_idx)
                    exploration_bonus = torch.zeros_like(q_values)
                    for action_idx in range(len(self.action_to_idx)):
                        action_name = self.idx_to_action[action_idx]
                        action_count = self._action_counts.get(action_name, 0) + 1
                        exploration_bonus[action_idx] = np.sqrt(2 * np.log(total_steps) / action_count)
                    
                    # Combine Q-values with exploration bonus
                    combined_values = q_values + 0.1 * exploration_bonus
                else:
                    combined_values = q_values
                
                action_idx = torch.argmax(combined_values).item()
                confidence = float(torch.max(torch.softmax(q_values, dim=0)))
        
        # Update action statistics
        action_name = self.idx_to_action[action_idx]
        if not hasattr(self, '_action_counts'):
            self._action_counts = {}
        if not hasattr(self, '_action_success_count'):
            self._action_success_count = {}
        
        self._action_counts[action_name] = self._action_counts.get(action_name, 0) + 1
        
        return action_name, confidence
    
    def update_action_success(self, action_name: str, reward: float):
        """Update action success statistics"""
        if not hasattr(self, '_action_success_count'):
            self._action_success_count = {}
        
        # Consider an action successful if reward is above average
        if reward > 0.3:  # Threshold for success
            self._action_success_count[action_name] = self._action_success_count.get(action_name, 0) + 1
    
    def learn(self, experience: Experience) -> Dict[str, float]:
        """Learn from an experience"""
        # Update action success statistics
        self.update_action_success(experience.action, experience.reward)
        
        # Convert action name to index
        action_idx = self.action_to_idx.get(experience.action, 0)
        
        # Store experience with index
        modified_experience = Experience(
            state=experience.state.detach().float(),
            action=str(action_idx),  # Store as string to maintain compatibility
            reward=float(experience.reward),
            next_state=experience.next_state.detach().float(),
            done=experience.done
        )
        self.experience_buffer.add(modified_experience)
        
        # Only start learning when we have enough experiences
        if len(self.experience_buffer) < self.batch_size:
            return {'loss': 0.0, 'avg_q_value': 0.0, 'max_q_value': 0.0}
        
        # Sample batch of experiences
        batch = self.experience_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([exp.state for exp in batch]).float()
        next_states = torch.stack([exp.next_state for exp in batch]).float()
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32)
        action_indices = torch.tensor([int(exp.action) for exp in batch], dtype=torch.int64)
        
        # Current Q-values
        current_q = self.q_network(states)
        current_q_values = current_q.gather(1, action_indices.unsqueeze(1)).squeeze()
        
        # Target Q-values with double Q-learning
        with torch.no_grad():
            # Select actions using online network
            next_q_online = self.q_network(next_states)
            next_actions = torch.argmax(next_q_online, dim=1)
            
            # Evaluate actions using target network
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss with Huber loss for stability
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Detach values for metrics
        return {
            'loss': float(loss.detach()),
            'avg_q_value': float(torch.mean(current_q_values.detach())),
            'max_q_value': float(torch.max(current_q_values.detach()))
        }
    
    def update_exploration(self, decay_factor: float = 0.995) -> None:
        """Update exploration rate"""
        self.epsilon = max(0.01, self.epsilon * decay_factor)  # Minimum exploration of 1% 