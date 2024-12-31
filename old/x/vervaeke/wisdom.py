"""
Module for handling wisdom components based on Vervaeke's framework.
Implements mechanisms for managing wisdom development through virtue and relevance realization.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from .rationality_core import RationalityCore
from .mindset import MindsetCore
from .cognitive_core import CognitiveCore

class WisdomType(Enum):
    SOPHIA = "sophia"  # Theoretical wisdom
    PHRONESIS = "phronesis"  # Practical wisdom

class VirtueType(Enum):
    HONESTY = "honesty"
    KINDNESS = "kindness"
    COURAGE = "courage"
    JUSTICE = "justice"
    TEMPERANCE = "temperance"

class KnowingMode(Enum):
    PROPOSITIONAL = "propositional"
    PROCEDURAL = "procedural"
    PERSPECTIVAL = "perspectival"
    PARTICIPATORY = "participatory"

@dataclass
class WisdomCriterion:
    """Represents a criterion for wisdom based on Berlin Wisdom Paradigm."""
    factual_knowledge: float  # Rich factual knowledge about life pragmatics
    procedural_knowledge: float  # Rich procedural knowledge
    lifespan_contextualism: float  # Understanding life contexts
    value_relativism: float  # Understanding value diversity
    uncertainty_management: float  # Managing uncertainty

@dataclass
class VirtueContext:
    """Represents a context for virtue application."""
    relevant_virtues: Set[VirtueType]
    conflicts: List[tuple[VirtueType, VirtueType]]
    context_factors: Dict[str, float]
    optimal_balance: Dict[VirtueType, float]

class WisdomCore:
    """Core class for managing wisdom development."""
    
    def __init__(
        self,
        rationality_core: RationalityCore,
        mindset_core: MindsetCore,
        cognitive_core: CognitiveCore
    ):
        self.rationality_core = rationality_core
        self.mindset_core = mindset_core
        self.cognitive_core = cognitive_core
        self.wisdom_criteria = WisdomCriterion(
            factual_knowledge=0.0,
            procedural_knowledge=0.0,
            lifespan_contextualism=0.0,
            value_relativism=0.0,
            uncertainty_management=0.0
        )
        self.active_virtues: Dict[VirtueType, float] = {
            virtue: 0.0 for virtue in VirtueType
        }
        self.knowing_modes: Dict[KnowingMode, float] = {
            mode: 0.0 for mode in KnowingMode
        }
        
    def analyze_virtue_context(self, context: VirtueContext) -> Dict[VirtueType, float]:
        """Analyze a context to determine optimal virtue balance."""
        recommendations = {}
        for virtue in context.relevant_virtues:
            base_weight = context.optimal_balance.get(virtue, 0.0)
            conflict_adjustment = self._calculate_conflict_adjustment(
                virtue, context.conflicts
            )
            recommendations[virtue] = min(1.0, base_weight - conflict_adjustment)
        return recommendations
        
    def _calculate_conflict_adjustment(
        self,
        virtue: VirtueType,
        conflicts: List[tuple[VirtueType, VirtueType]]
    ) -> float:
        """Calculate adjustment needed due to virtue conflicts."""
        adjustment = 0.0
        for v1, v2 in conflicts:
            if virtue in (v1, v2):
                adjustment += 0.1  # Reduce weight when in conflict
        return min(0.5, adjustment)  # Cap adjustment at 0.5
        
    def evaluate_wisdom(self) -> Dict[str, float]:
        """Evaluate overall wisdom development."""
        rationality_score = self.rationality_core.evaluate_rationality()["overall_rationality"]
        mindset_impact = self.mindset_core.evaluate_mindset_impact()
        
        sophia_score = (
            self.wisdom_criteria.factual_knowledge * 0.4 +
            self.wisdom_criteria.uncertainty_management * 0.3 +
            self.knowing_modes[KnowingMode.PROPOSITIONAL] * 0.3
        )
        
        phronesis_score = (
            self.wisdom_criteria.procedural_knowledge * 0.3 +
            self.wisdom_criteria.lifespan_contextualism * 0.3 +
            self.knowing_modes[KnowingMode.PROCEDURAL] * 0.2 +
            self.knowing_modes[KnowingMode.PERSPECTIVAL] * 0.2
        )
        
        return {
            "sophia": sophia_score,
            "phronesis": phronesis_score,
            "overall_wisdom": (sophia_score + phronesis_score) / 2,
            "rationality_contribution": rationality_score * 0.3,
            "mindset_contribution": mindset_impact["effective_rationality"] * 0.2,
            "virtue_development": sum(self.active_virtues.values()) / len(VirtueType)
        }
        
    def optimize_wisdom(self) -> List[str]:
        """Generate recommendations for optimizing wisdom."""
        recommendations = []
        
        # Check knowing mode balance
        weak_modes = [
            mode for mode, level in self.knowing_modes.items()
            if level < 0.3
        ]
        if weak_modes:
            recommendations.append(
                f"Develop knowing modes: {', '.join(m.value for m in weak_modes)}"
            )
            
        # Check wisdom criteria
        criteria_dict = self.wisdom_criteria.__dict__
        weak_criteria = [
            criterion for criterion, level in criteria_dict.items()
            if level < 0.3
        ]
        if weak_criteria:
            recommendations.append(
                f"Strengthen wisdom criteria: {', '.join(weak_criteria)}"
            )
            
        # Check virtue development
        weak_virtues = [
            virtue for virtue, level in self.active_virtues.items()
            if level < 0.3
        ]
        if weak_virtues:
            recommendations.append(
                f"Cultivate virtues: {', '.join(v.value for v in weak_virtues)}"
            )
            
        return recommendations 