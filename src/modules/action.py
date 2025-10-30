import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class Action:
    """Represents an action to be taken by the system"""
    name: str
    parameters: Dict[str, Any]
    confidence: float
    expected_outcome: torch.Tensor
    priority: float
    execution_cost: float = 1.0
    temporal_constraint: Optional[float] = None


@dataclass
class Goal:
    """Represents a hierarchical goal"""
    name: str
    description: str
    priority: float
    deadline: Optional[float] = None
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    completion_status: float = 0.0  # 0.0 to 1.0
    required_actions: List[str] = field(default_factory=list)


class GoalHierarchy:
    """Manages hierarchical goal structures with temporal constraints"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goals: List[str] = []
        self.completed_goals: List[str] = []
    
    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the hierarchy"""
        self.goals[goal.name] = goal
        if goal.parent_goal and goal.parent_goal in self.goals:
            self.goals[goal.parent_goal].subgoals.append(goal.name)
        if goal.name not in self.active_goals:
            self.active_goals.append(goal.name)
    
    def update_goal_progress(self, goal_name: str, progress: float) -> None:
        """Update progress on a goal"""
        if goal_name in self.goals:
            self.goals[goal_name].completion_status = min(1.0, progress)
            if self.goals[goal_name].completion_status >= 1.0:
                self.complete_goal(goal_name)
    
    def complete_goal(self, goal_name: str) -> None:
        """Mark a goal as completed"""
        if goal_name in self.active_goals:
            self.active_goals.remove(goal_name)
            self.completed_goals.append(goal_name)
            
            # Update parent goal progress
            goal = self.goals[goal_name]
            if goal.parent_goal and goal.parent_goal in self.goals:
                parent = self.goals[goal.parent_goal]
                completed_subgoals = sum(
                    1 for sg in parent.subgoals 
                    if sg in self.completed_goals
                )
                parent.completion_status = completed_subgoals / len(parent.subgoals)
    
    def get_active_goals_sorted(self) -> List[Goal]:
        """Get active goals sorted by priority and deadline"""
        current_time = time.time()
        
        def goal_urgency(goal_name: str) -> float:
            goal = self.goals[goal_name]
            urgency = goal.priority
            
            # Boost urgency if deadline is approaching
            if goal.deadline is not None:
                time_remaining = max(0, goal.deadline - current_time)
                urgency += 2.0 / (1.0 + time_remaining)
            
            return urgency
        
        sorted_goals = sorted(
            self.active_goals,
            key=goal_urgency,
            reverse=True
        )
        
        return [self.goals[g] for g in sorted_goals]
    
    def get_next_required_actions(self, top_k: int = 3) -> List[str]:
        """Get the next required actions from top priority goals"""
        active_goals = self.get_active_goals_sorted()[:top_k]
        actions = []
        for goal in active_goals:
            actions.extend(goal.required_actions)
        return list(set(actions))  # Remove duplicates

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

class DynamicResourceAllocator:
    """Manages dynamic resource allocation for action execution"""
    
    def __init__(self, total_resources: float = 100.0):
        self.total_resources = total_resources
        self.available_resources = total_resources
        self.allocated_resources: Dict[str, float] = {}
        self.allocation_history: List[Dict[str, float]] = []
    
    def allocate(self, action_name: str, amount: float) -> bool:
        """Allocate resources for an action"""
        if amount <= self.available_resources:
            self.allocated_resources[action_name] = amount
            self.available_resources -= amount
            self.allocation_history.append(self.allocated_resources.copy())
            return True
        return False
    
    def release(self, action_name: str) -> None:
        """Release allocated resources"""
        if action_name in self.allocated_resources:
            self.available_resources += self.allocated_resources[action_name]
            del self.allocated_resources[action_name]
    
    def get_utilization(self) -> float:
        """Get current resource utilization rate"""
        return 1.0 - (self.available_resources / self.total_resources)
    
    def optimize_allocation(self, action_priorities: Dict[str, float]) -> Dict[str, float]:
        """Optimize resource allocation based on action priorities"""
        total_priority = sum(action_priorities.values())
        if total_priority == 0:
            return {}
        
        optimal_allocation = {}
        for action, priority in action_priorities.items():
            allocation = (priority / total_priority) * self.available_resources
            optimal_allocation[action] = allocation
        
        return optimal_allocation


class ActionSelectionModule(nn.Module):
    """Enhanced action selection with planning and resource management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.feature_dim = self.config.get('feature_dim', 512)
        self.action_repertoire = ActionRepertoire()
        self.goal_hierarchy = GoalHierarchy()
        self.resource_allocator = DynamicResourceAllocator(
            total_resources=self.config.get('total_resources', 100.0)
        )
        
        # Enhanced policy networks
        self.goal_processor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        self.action_policy = nn.Sequential(
            nn.Linear(128 + self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.outcome_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.Tanh()
        )
        
        # Enhanced priority network with resource awareness
        self.priority_network = nn.Sequential(
            nn.Linear(self.feature_dim + 128 + 1, 64),  # +1 for resource availability
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Cost estimator for resource allocation
        self.cost_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Error correction network
        self.error_corrector = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),  # Expected vs actual outcome
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
            nn.Tanh()
        )
        
        # Performance tracking
        self.action_success_rates: Dict[str, List[float]] = {}
        self.error_recovery_count = 0
    
    def _process_goals(self, 
                      current_thought: torch.Tensor,
                      goal_stack: List[str]) -> torch.Tensor:
        """Process current goals with thought context"""
        # Use goal hierarchy to get encoded goals
        goal_encoding = torch.zeros(self.feature_dim)
        
        if goal_stack or len(self.goal_hierarchy.active_goals) > 0:
            # Encode goals based on their priorities and contexts
            # Simple hash-based encoding for now
            for i, goal in enumerate(goal_stack[:5]):  # Top 5 goals
                goal_hash = hash(goal) % self.feature_dim
                goal_encoding[goal_hash % self.feature_dim] += 0.2 * (5 - i) / 5
        
        return self.goal_processor(goal_encoding)
    
    def _generate_action_candidates(self, 
                                  thought_features: torch.Tensor,
                                  goal_features: torch.Tensor,
                                  resource_availability: float) -> List[Action]:
        """Generate action candidates with resource and goal awareness"""
        combined_features = torch.cat([thought_features, goal_features], dim=-1)
        action_features = self.action_policy(combined_features)
        
        # Predict outcomes
        predicted_outcome = self.outcome_predictor(action_features)
        
        # Estimate execution cost
        estimated_cost = float(self.cost_estimator(action_features)) * 10.0  # Scale to resource units
        
        # Calculate priority with resource awareness
        resource_tensor = torch.tensor([resource_availability])
        priority_input = torch.cat([predicted_outcome, action_features, resource_tensor], dim=-1)
        priority = float(self.priority_network(priority_input))
        
        # Get required actions from goal hierarchy
        required_actions = self.goal_hierarchy.get_next_required_actions()
        
        # Generate candidate actions
        candidates = []
        for action_name in self.action_repertoire.actions:
            # Boost priority if action is required by goals
            priority_boost = 1.5 if action_name in required_actions else 1.0
            
            if action_name == 'focus_attention':
                action = Action(
                    name=action_name,
                    parameters={
                        'target_modality': 'visual',
                        'target_location': [0.5, 0.5]
                    },
                    confidence=float(torch.sigmoid(torch.rand(1))),
                    expected_outcome=predicted_outcome,
                    priority=priority * priority_boost,
                    execution_cost=estimated_cost * 0.5
                )
            elif action_name == 'external_action':
                action = Action(
                    name=action_name,
                    parameters={
                        'action_type': 'explore',
                        'action_params': {'intensity': 0.8}
                    },
                    confidence=float(torch.sigmoid(torch.rand(1))),
                    expected_outcome=predicted_outcome,
                    priority=priority * priority_boost,
                    execution_cost=estimated_cost * 1.2
                )
            else:
                action = Action(
                    name=action_name,
                    parameters={
                        'default': 'value',
                        'intensity': 0.5
                    },
                    confidence=float(torch.sigmoid(torch.rand(1))),
                    expected_outcome=predicted_outcome,
                    priority=priority * priority_boost,
                    execution_cost=estimated_cost
                )
            
            # Only include if we can afford it
            if self.resource_allocator.available_resources >= action.execution_cost:
                if self.action_repertoire.validate_action(action):
                    candidates.append(action)
        
        return candidates
    
    def recover_from_error(self, 
                          expected_outcome: torch.Tensor,
                          actual_outcome: torch.Tensor) -> torch.Tensor:
        """Generate error correction signal"""
        error_input = torch.cat([expected_outcome, actual_outcome], dim=-1)
        correction = self.error_corrector(error_input)
        self.error_recovery_count += 1
        return correction
    
    def update_action_success(self, action_name: str, success: bool) -> None:
        """Update success tracking for an action"""
        if action_name not in self.action_success_rates:
            self.action_success_rates[action_name] = []
        self.action_success_rates[action_name].append(1.0 if success else 0.0)
    
    def select_action(self,
                     current_thought: torch.Tensor,
                     goal_stack: List[str],
                     emotional_valence: float) -> Optional[Action]:
        """Select action with enhanced planning and resource management"""
        
        # Process goals with thought context
        goal_features = self._process_goals(current_thought, goal_stack)
        
        # Get resource availability
        resource_availability = self.resource_allocator.available_resources / 100.0
        
        # Generate candidate actions
        candidates = self._generate_action_candidates(
            current_thought, goal_features, resource_availability
        )
        
        if not candidates:
            return None
        
        # Select action with resource-aware priority and emotional modulation
        modulated_priorities = []
        for action in candidates:
            # Consider success history if available
            success_rate = 1.0
            if action.name in self.action_success_rates:
                recent_successes = self.action_success_rates[action.name][-10:]
                success_rate = np.mean(recent_successes) if recent_successes else 1.0
            
            # Modulate priority by emotion, success rate, and resource efficiency
            resource_efficiency = 1.0 / (action.execution_cost + 1.0)
            final_priority = (
                action.priority * 
                (1 + emotional_valence * 0.5) *  # Emotional influence
                success_rate *  # Historical success
                (1 + 0.3 * resource_efficiency)  # Resource efficiency bonus
            )
            
            modulated_priorities.append((final_priority, action))
        
        # Return highest priority action
        _, selected_action = max(modulated_priorities, key=lambda x: x[0])
        
        # Allocate resources
        self.resource_allocator.allocate(selected_action.name, selected_action.execution_cost)
        
        return selected_action
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get planning and resource management metrics"""
        metrics = {
            'resource_utilization': self.resource_allocator.get_utilization(),
            'active_goals': len(self.goal_hierarchy.active_goals),
            'completed_goals': len(self.goal_hierarchy.completed_goals),
            'error_recoveries': self.error_recovery_count,
            'action_success_rates': {
                name: np.mean(rates[-20:]) if rates else 0.0
                for name, rates in self.action_success_rates.items()
            }
        }
        return metrics
    
    def forward(self,
               current_thought: torch.Tensor,
               goal_stack: List[str],
               emotional_valence: float) -> Optional[Action]:
        """Forward pass of enhanced action selection module"""
        return self.select_action(current_thought, goal_stack, emotional_valence)
        return self.select_action(current_thought, goal_stack, emotional_valence) 