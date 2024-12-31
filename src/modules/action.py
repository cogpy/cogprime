import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Action:
    """Represents an action to be taken by the system"""
    name: str
    parameters: Dict[str, Any]
    confidence: float
    expected_outcome: torch.Tensor
    priority: float

class ActionRepertoire:
    """Manages the available actions and their prerequisites"""
    
    def __init__(self):
        self.actions = {
            'focus_attention': {
                'description': 'Direct attention to specific sensory input',
                'parameters': ['target_modality', 'target_location']
            },
            'update_goal': {
                'description': 'Modify current goal stack',
                'parameters': ['goal_operation', 'goal_content']
            },
            'query_memory': {
                'description': 'Explicitly query episodic memory',
                'parameters': ['query_content', 'context']
            },
            'external_action': {
                'description': 'Execute action in external environment',
                'parameters': ['action_type', 'action_params']
            }
        }
    
    def validate_action(self, action: Action) -> bool:
        """Validate if action is well-formed and executable"""
        if action.name not in self.actions:
            return False
        required_params = set(self.actions[action.name]['parameters'])
        provided_params = set(action.parameters.keys())
        return required_params.issubset(provided_params)

class ActionSelectionModule(nn.Module):
    """Implements action selection and planning mechanisms"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.feature_dim = self.config.get('feature_dim', 512)
        self.action_repertoire = ActionRepertoire()
        
        # Policy networks
        self.goal_processor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.action_policy = nn.Sequential(
            nn.Linear(128 + self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.outcome_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.Tanh()
        )
        
        # Action priority network
        self.priority_network = nn.Sequential(
            nn.Linear(self.feature_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _process_goals(self, 
                      current_thought: torch.Tensor,
                      goal_stack: List[str]) -> torch.Tensor:
        """Process current goals with thought context"""
        # Encode goals into tensor (placeholder for more sophisticated encoding)
        goal_encoding = torch.zeros(self.feature_dim)
        if goal_stack:
            # Simple averaging of goal representations
            goal_encoding = torch.randn(self.feature_dim)  # Placeholder
        return self.goal_processor(goal_encoding)
    
    def _generate_action_candidates(self, 
                                  thought_features: torch.Tensor,
                                  goal_features: torch.Tensor) -> List[Action]:
        """Generate potential actions based on current thought and goals"""
        combined_features = torch.cat([thought_features, goal_features], dim=-1)
        action_features = self.action_policy(combined_features)
        
        # Predict outcomes for different action types
        predicted_outcome = self.outcome_predictor(action_features)
        
        # Calculate action priority
        priority = self.priority_network(
            torch.cat([predicted_outcome, action_features], dim=-1)
        )
        
        # Generate candidate actions with specific parameters
        candidates = []
        for action_name in self.action_repertoire.actions:
            if action_name == 'focus_attention':
                action = Action(
                    name=action_name,
                    parameters={
                        'target_modality': 'visual',
                        'target_location': [0.5, 0.5]  # Center of attention
                    },
                    confidence=float(torch.rand(1)),
                    expected_outcome=predicted_outcome,
                    priority=float(priority)
                )
            elif action_name == 'external_action':
                action = Action(
                    name=action_name,
                    parameters={
                        'action_type': 'explore',
                        'action_params': {'intensity': 0.8}
                    },
                    confidence=float(torch.rand(1)),
                    expected_outcome=predicted_outcome,
                    priority=float(priority)
                )
            else:
                action = Action(
                    name=action_name,
                    parameters={
                        'default': 'value',
                        'intensity': 0.5
                    },
                    confidence=float(torch.rand(1)),
                    expected_outcome=predicted_outcome,
                    priority=float(priority)
                )
            
            if self.action_repertoire.validate_action(action):
                candidates.append(action)
        
        return candidates
    
    def select_action(self,
                     current_thought: torch.Tensor,
                     goal_stack: List[str],
                     emotional_valence: float) -> Optional[Action]:
        """Select the most appropriate action given current cognitive state"""
        
        # Process goals in context of current thought
        goal_features = self._process_goals(current_thought, goal_stack)
        
        # Generate candidate actions
        candidates = self._generate_action_candidates(current_thought, goal_features)
        
        if not candidates:
            return None
            
        # Select action based on priority and emotional modulation
        modulated_priorities = [
            (action.priority * (1 + emotional_valence), action)
            for action in candidates
        ]
        
        # Return highest priority action
        _, selected_action = max(modulated_priorities, key=lambda x: x[0])
        return selected_action
    
    def forward(self,
               current_thought: torch.Tensor,
               goal_stack: List[str],
               emotional_valence: float) -> Optional[Action]:
        """Forward pass of action selection module"""
        return self.select_action(current_thought, goal_stack, emotional_valence) 