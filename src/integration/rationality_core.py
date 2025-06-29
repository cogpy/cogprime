"""
Module for handling rationality components based on Vervaeke's framework.
Implements mechanisms for systematic overcoming of self-deception through active open-mindedness.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from ..core.cognitive_core import CognitiveCore
from .relevance_core import RelevanceCore

class BiasType(Enum):
    CONFIRMATION = "confirmation"
    ESSENTIALISM = "essentialism"
    AVAILABILITY = "availability"
    ANCHORING = "anchoring"
    FRAMING = "framing"

class CognitionNeed(Enum):
    CURIOSITY = "curiosity"
    WONDER = "wonder"
    PROBLEM_SOLVING = "problem_solving"
    WORLDVIEW_QUESTIONING = "worldview_questioning"

@dataclass
class Bias:
    """Represents a cognitive bias."""
    type: BiasType
    description: str
    impact: float
    frequency: float
    counteraction_strategy: str

@dataclass
class CognitionStyle:
    """Represents a cognitive style."""
    active_open_mindedness: float
    need_for_cognition: Dict[CognitionNeed, float]
    bias_awareness: Dict[BiasType, float]
    counteraction_effectiveness: Dict[BiasType, float]

class RationalityCore:
    """Core class for managing rationality development."""
    
    def __init__(self, cognitive_core: CognitiveCore, relevance_core: RelevanceCore):
        self.cognitive_core = cognitive_core
        self.relevance_core = relevance_core
        self.biases: Dict[BiasType, Bias] = {}
        self.cognition_style = CognitionStyle(
            active_open_mindedness=0.0,
            need_for_cognition={need: 0.0 for need in CognitionNeed},
            bias_awareness={bias: 0.0 for bias in BiasType},
            counteraction_effectiveness={bias: 0.0 for bias in BiasType}
        )
        
    def register_bias(self, bias: Bias) -> None:
        """Register a cognitive bias."""
        self.biases[bias.type] = bias
        self._update_bias_awareness(bias.type, bias.frequency)
        
    def cultivate_open_mindedness(self, bias_type: BiasType, success: bool) -> None:
        """Cultivate active open-mindedness through bias counteraction."""
        if success:
            self.cognition_style.counteraction_effectiveness[bias_type] += 0.1
            self.cognition_style.active_open_mindedness = min(
                1.0,
                sum(self.cognition_style.counteraction_effectiveness.values()) / len(BiasType)
            )
        
    def develop_cognition_need(self, need: CognitionNeed, intensity: float) -> None:
        """Develop need for cognition in a specific area."""
        self.cognition_style.need_for_cognition[need] = min(
            1.0,
            self.cognition_style.need_for_cognition[need] + intensity
        )
        
    def _update_bias_awareness(self, bias_type: BiasType, frequency: float) -> None:
        """Update awareness of a specific bias."""
        self.cognition_style.bias_awareness[bias_type] = min(
            1.0,
            self.cognition_style.bias_awareness[bias_type] + frequency
        )
        
    def evaluate_rationality(self) -> Dict[str, float]:
        """Evaluate overall rationality development."""
        intelligence_score = self.cognitive_core.evaluate_intelligence()
        relevance_score = self.relevance_core.evaluate_relevance()
        
        open_mindedness = self.cognition_style.active_open_mindedness
        cognition_need = sum(self.cognition_style.need_for_cognition.values()) / len(CognitionNeed)
        bias_management = sum(self.cognition_style.counteraction_effectiveness.values()) / len(BiasType)
        
        return {
            "intelligence_contribution": intelligence_score * 0.3,  # Based on Stanovich's findings
            "relevance_contribution": relevance_score * 0.2,
            "open_mindedness_contribution": open_mindedness * 0.2,
            "cognition_need_contribution": cognition_need * 0.15,
            "bias_management_contribution": bias_management * 0.15,
            "overall_rationality": sum([
                intelligence_score * 0.3,
                relevance_score * 0.2,
                open_mindedness * 0.2,
                cognition_need * 0.15,
                bias_management * 0.15
            ])
        }
        
    def optimize_rationality(self) -> List[str]:
        """Generate recommendations for optimizing rationality."""
        recommendations = []
        
        # Check for low bias awareness
        weak_awareness = [
            bias for bias, awareness in self.cognition_style.bias_awareness.items()
            if awareness < 0.3
        ]
        if weak_awareness:
            recommendations.append(
                f"Increase awareness of biases: {', '.join(b.value for b in weak_awareness)}"
            )
            
        # Check for low counteraction effectiveness
        weak_counteraction = [
            bias for bias, effectiveness in self.cognition_style.counteraction_effectiveness.items()
            if effectiveness < 0.3
        ]
        if weak_counteraction:
            recommendations.append(
                f"Improve counteraction strategies for: {', '.join(b.value for b in weak_counteraction)}"
            )
            
        # Check for underdeveloped cognition needs
        weak_needs = [
            need for need, level in self.cognition_style.need_for_cognition.items()
            if level < 0.3
        ]
        if weak_needs:
            recommendations.append(
                f"Cultivate stronger need for: {', '.join(n.value for n in weak_needs)}"
            )
            
        return recommendations 