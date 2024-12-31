import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ..modules.perception import PerceptionModule, SensoryInput
from ..modules.reasoning import ReasoningModule, Thought
from ..modules.action import ActionSelectionModule, Action
from ..modules.learning import ReinforcementLearner, Experience

@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system"""
    attention_focus: torch.Tensor
    working_memory: Dict[str, Any]
    emotional_valence: float
    goal_stack: List[str]
    sensory_buffer: Dict[str, torch.Tensor]
    current_thought: Thought = None
    last_action: Action = None
    last_reward: float = 0.0
    total_reward: float = 0.0

class CogPrimeCore:
    """
    The core cognitive architecture of CogPrime system.
    Implements the basic cognitive cycle and main AGI components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.perception = PerceptionModule(config)
        self.reasoning = ReasoningModule(config)
        self.action_selector = ActionSelectionModule(config)
        self.learner = ReinforcementLearner(config)
        self.state = CognitiveState(
            attention_focus=torch.zeros(512),  # Initial attention vector
            working_memory={},
            emotional_valence=0.0,
            goal_stack=[],
            sensory_buffer={},
            current_thought=None,
            last_action=None,
            last_reward=0.0,
            total_reward=0.0
        )
        
    def cognitive_cycle(self, sensory_input: SensoryInput, reward: float = 0.0) -> Optional[Action]:
        """Execute one cognitive cycle with learning"""
        # Store current state for learning
        current_state = self.state.attention_focus
        
        # Process cycle
        self._perceive(sensory_input)
        self._reason()
        action = self._act()
        
        # Update rewards
        self.state.last_reward = reward
        self.state.total_reward += reward
        
        # Learn from experience if we have a previous action
        if action and self.state.last_action:
            experience = Experience(
                state=current_state,
                action=self.state.last_action.name,
                reward=reward,
                next_state=self.state.attention_focus,
                done=False  # Could be based on goal achievement
            )
            
            # Update learning system
            learning_stats = self.learner.learn(experience)
            self.state.working_memory['learning_stats'] = learning_stats
            
            # Update exploration rate
            self.learner.update_exploration()
        
        return action
    
    def _perceive(self, sensory_input: SensoryInput) -> None:
        """Perception phase of the cognitive cycle"""
        # Process sensory input through perception module
        attended_features, attention_weights = self.perception.process_input(sensory_input)
        
        # Update cognitive state
        self.state.attention_focus = attention_weights
        self.state.sensory_buffer = {
            'attended_features': attended_features,
            'raw_input': sensory_input
        }
    
    def _reason(self) -> None:
        """Reasoning phase of the cognitive cycle"""
        # Get attended features from sensory buffer
        attended_features = self.state.sensory_buffer['attended_features']
        
        # Process through reasoning module
        thought, updated_memory = self.reasoning(
            attended_features,
            self.state.working_memory
        )
        
        # Update cognitive state
        self.state.current_thought = thought
        self.state.working_memory = updated_memory
        
        # Update emotional valence based on thought salience and rewards
        self.state.emotional_valence = (
            self.state.emotional_valence * 0.7 +  # Decay factor
            thought.salience * 0.2 +  # Thought contribution
            np.tanh(self.state.last_reward) * 0.1  # Reward contribution
        )
    
    def _act(self) -> Optional[Action]:
        """Action phase of the cognitive cycle with learning influence"""
        if self.state.current_thought is None:
            return None
            
        # Get action suggestion from learner
        learner_action, confidence = self.learner.select_action(
            self.state.current_thought.content
        )
        
        # Combine with action selector
        selected_action = self.action_selector(
            self.state.current_thought.content,
            self.state.goal_stack,
            self.state.emotional_valence
        )
        
        # Use learner's suggestion if confidence is high enough
        if selected_action and confidence > 0.8:
            selected_action.name = learner_action
            selected_action.confidence = confidence
        
        # Update cognitive state
        self.state.last_action = selected_action
        
        return selected_action
    
    def update_goals(self, new_goal: str) -> None:
        """Update the system's goal stack"""
        self.state.goal_stack.append(new_goal)
    
    def get_cognitive_state(self) -> CognitiveState:
        """Return current cognitive state"""
        return self.state 