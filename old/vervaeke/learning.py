"""
Learning module for adapting relevance realization mechanisms.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from .cognitive_core import CognitiveFrame, KnowingMode

@dataclass
class RelevanceExperience:
    """Represents an experience of what was relevant in a context"""
    frame: CognitiveFrame
    inputs: Dict[str, Any]
    actual_relevance: Dict[str, float]
    reward: float

class RelevanceLearner:
    """Learns to adapt relevance realization based on experience"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.experiences: List[RelevanceExperience] = []
        
    def learn(self, experience: RelevanceExperience) -> Dict[str, float]:
        """Learn from a relevance realization experience"""
        self.experiences.append(experience)
        
        # Update salience weights based on actual relevance
        updates = {}
        for key, actual_value in experience.actual_relevance.items():
            if key in experience.frame.salience_weights:
                current = experience.frame.salience_weights[key]
                error = actual_value - current
                update = self.learning_rate * error * experience.reward
                experience.frame.salience_weights[key] += update
                updates[key] = update
                
        return updates
        
    def suggest_frame(self, inputs: Dict[str, Any]) -> CognitiveFrame:
        """Suggest a cognitive frame based on past experiences"""
        if not self.experiences:
            # Return default frame if no experiences
            return CognitiveFrame(
                salience_weights={'default': 1.0},
                active_knowing_modes=[KnowingMode.PROPOSITIONAL],
                context={}
            )
            
        # Find most similar past experience
        best_match = max(
            self.experiences,
            key=lambda exp: self._compute_similarity(exp.inputs, inputs)
        )
        
        # Create new frame based on best matching experience
        return CognitiveFrame(
            salience_weights=best_match.frame.salience_weights.copy(),
            active_knowing_modes=best_match.frame.active_knowing_modes.copy(),
            context=inputs
        )
        
    def _compute_similarity(self, exp_inputs: Dict[str, Any], 
                          current_inputs: Dict[str, Any]) -> float:
        """Compute similarity between two input sets"""
        # Simple overlap score for now
        common_keys = set(exp_inputs.keys()) & set(current_inputs.keys())
        if not common_keys:
            return 0.0
            
        matches = sum(1 for k in common_keys 
                     if exp_inputs[k] == current_inputs[k])
        return matches / len(common_keys) 