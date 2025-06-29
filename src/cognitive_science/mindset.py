"""
Module for handling mindset components based on Vervaeke's framework.
Implements mechanisms for managing cognitive styles and their recursive improvement.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from .rationality_core import RationalityCore
from ..core.cognitive_core import CognitiveCore

class MindsetType(Enum):
    FIXED = "fixed"
    MALLEABLE = "malleable"

class ProcessingStyle(Enum):
    S1_INTUITIVE = "s1_intuitive"
    S2_DELIBERATE = "s2_deliberate"

class ErrorResponse(Enum):
    PERMANENT_REVELATION = "permanent_revelation"
    SKILL_IMPROVEMENT = "skill_improvement"
    EFFORT_ADJUSTMENT = "effort_adjustment"

@dataclass
class CognitiveState:
    """Represents a cognitive state with processing styles."""
    s1_activation: float  # 0.0 to 1.0
    s2_activation: float  # 0.0 to 1.0
    current_context: str
    active_mindset: MindsetType

@dataclass
class ProcessingOutcome:
    """Represents the outcome of cognitive processing."""
    success: bool
    error_type: Optional[str]
    response: ErrorResponse
    learning_potential: float
    confidence: float

class MindsetCore:
    """Core class for managing mindset and cognitive styles."""
    
    def __init__(self, rationality_core: RationalityCore, cognitive_core: CognitiveCore):
        self.rationality_core = rationality_core
        self.cognitive_core = cognitive_core
        self.cognitive_state = CognitiveState(
            s1_activation=0.5,
            s2_activation=0.5,
            current_context="default",
            active_mindset=MindsetType.MALLEABLE
        )
        
    def set_context(self, context: str) -> None:
        """Set the current cognitive context."""
        self.cognitive_state.current_context = context
        self._adjust_processing_balance(context)
        
    def _adjust_processing_balance(self, context: str) -> None:
        """Adjust S1/S2 balance based on context."""
        if context == "theorizing":
            self.cognitive_state.s2_activation = 0.8
            self.cognitive_state.s1_activation = 0.2
        elif context == "therapy":
            self.cognitive_state.s2_activation = 0.2
            self.cognitive_state.s1_activation = 0.8
        else:
            self.cognitive_state.s2_activation = 0.5
            self.cognitive_state.s1_activation = 0.5
            
    def process_outcome(self, outcome: ProcessingOutcome) -> None:
        """Process an outcome and adjust mindset accordingly."""
        if self.cognitive_state.active_mindset == MindsetType.FIXED:
            if not outcome.success:
                self._handle_fixed_mindset_error(outcome)
        else:
            self._handle_malleable_mindset_outcome(outcome)
            
    def _handle_fixed_mindset_error(self, outcome: ProcessingOutcome) -> None:
        """Handle error in fixed mindset context."""
        # Fixed mindset tends to see errors as permanent revelations
        outcome.response = ErrorResponse.PERMANENT_REVELATION
        outcome.learning_potential = 0.1
        outcome.confidence *= 0.5
        
    def _handle_malleable_mindset_outcome(self, outcome: ProcessingOutcome) -> None:
        """Handle outcome in malleable mindset context."""
        if not outcome.success:
            if outcome.error_type == "skill_related":
                outcome.response = ErrorResponse.SKILL_IMPROVEMENT
                outcome.learning_potential = 0.8
            else:
                outcome.response = ErrorResponse.EFFORT_ADJUSTMENT
                outcome.learning_potential = 0.7
            # Malleable mindset maintains confidence despite errors
            outcome.confidence *= 0.9
            
    def evaluate_mindset_impact(self) -> Dict[str, float]:
        """Evaluate the impact of current mindset on cognitive performance."""
        intelligence_score = self.cognitive_core.evaluate_intelligence()
        rationality_score = self.rationality_core.evaluate_rationality()["overall_rationality"]
        
        mindset_factor = 1.2 if self.cognitive_state.active_mindset == MindsetType.MALLEABLE else 0.8
        
        return {
            "base_intelligence": intelligence_score,
            "effective_intelligence": intelligence_score * mindset_factor,
            "base_rationality": rationality_score,
            "effective_rationality": rationality_score * mindset_factor,
            "learning_potential": 0.8 if self.cognitive_state.active_mindset == MindsetType.MALLEABLE else 0.3,
            "error_recovery": 0.9 if self.cognitive_state.active_mindset == MindsetType.MALLEABLE else 0.4
        }
        
    def optimize_processing(self) -> List[str]:
        """Generate recommendations for optimizing cognitive processing."""
        recommendations = []
        
        # Check processing balance
        if self.cognitive_state.current_context == "theorizing":
            if self.cognitive_state.s2_activation < 0.7:
                recommendations.append("Increase deliberate processing for theoretical work")
        elif self.cognitive_state.current_context == "therapy":
            if self.cognitive_state.s1_activation < 0.7:
                recommendations.append("Increase intuitive processing for therapeutic work")
                
        # Check mindset alignment
        if self.cognitive_state.active_mindset == MindsetType.FIXED:
            recommendations.append("Consider shifting to a more malleable mindset for better learning")
            
        # Check processing effectiveness
        state_eval = self.evaluate_mindset_impact()
        if state_eval["effective_intelligence"] < state_eval["base_intelligence"]:
            recommendations.append("Current mindset is limiting cognitive potential")
            
        return recommendations 