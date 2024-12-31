"""
Module for handling enlightenment components based on Vervaeke's framework.
Implements mechanisms for addressing perennial problems through counteractive dynamical systems.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

class PerennialProblem(Enum):
    PARASITIC_PROCESSING = "parasitic_processing"
    MODAL_CONFUSION = "modal_confusion" 
    REFLECTIVENESS_GAP = "reflectiveness_gap"
    ABSURDITY = "absurdity"
    ANXIETY = "anxiety"
    ALIENATION = "alienation"
    EXISTENTIAL_INERTIA = "existential_inertia"
    EXISTENTIAL_IGNORANCE = "existential_ignorance"
    EXISTENTIAL_ENTRAPMENT = "existential_entrapment"

class PracticeType(Enum):
    COUNTERACTIVE_DYNAMICS = "counteractive_dynamics"
    SATI = "sati"
    FLOW = "flow"
    SCIENTIA_INTUITIVA = "scientia_intuitiva"
    INNER_DIALOGUE = "inner_dialogue"
    COMMUNITAS = "communitas"
    GNOSIS = "gnosis"
    WISDOM = "wisdom"

@dataclass
class Practice:
    """Represents a specific practice for addressing perennial problems."""
    type: PracticeType
    target_problems: Set[PerennialProblem]
    description: str
    prerequisites: Set[PracticeType]
    effects: Dict[str, float]  # Effect name -> strength

class EnlightenmentCore:
    """Core class for managing enlightenment practices and their effects."""
    
    def __init__(self):
        self.practices: Dict[PracticeType, Practice] = {}
        self.active_practices: Set[PracticeType] = set()
        self.problem_states: Dict[PerennialProblem, float] = {
            problem: 0.0 for problem in PerennialProblem
        }
        
    def add_practice(self, practice: Practice) -> None:
        """Add a practice to the available practices."""
        self.practices[practice.type] = practice
        
    def activate_practice(self, practice_type: PracticeType) -> bool:
        """Attempt to activate a practice if prerequisites are met."""
        practice = self.practices.get(practice_type)
        if not practice:
            return False
            
        if not practice.prerequisites.issubset(self.active_practices):
            return False
            
        self.active_practices.add(practice_type)
        self._apply_practice_effects(practice)
        return True
        
    def _apply_practice_effects(self, practice: Practice) -> None:
        """Apply the effects of a practice on perennial problems."""
        for problem in practice.target_problems:
            self.problem_states[problem] *= (1.0 - practice.effects.get("reduction", 0.0))
            
    def get_problem_state(self, problem: PerennialProblem) -> float:
        """Get the current state/severity of a perennial problem."""
        return self.problem_states[problem]
        
    def suggest_next_practice(self) -> Optional[PracticeType]:
        """Suggest the next practice to adopt based on current states."""
        max_problem = max(
            self.problem_states.items(),
            key=lambda x: x[1]
        )
        
        available_practices = [
            p for p in self.practices.values()
            if max_problem[0] in p.target_problems
            and p.type not in self.active_practices
            and p.prerequisites.issubset(self.active_practices)
        ]
        
        if not available_practices:
            return None
            
        return max(
            available_practices,
            key=lambda p: sum(p.effects.values())
        ).type

    def evaluate_enlightenment(self) -> float:
        """Evaluate overall progress towards enlightenment."""
        if not self.problem_states:
            return 0.0
            
        return 1.0 - (sum(self.problem_states.values()) / len(self.problem_states)) 