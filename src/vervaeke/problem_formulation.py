from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Callable
from enum import Enum
import numpy as np

from .relevance_core import RelevanceCore, RelevanceMode

class ProblemType(Enum):
    """Types of problems as described by Vervaeke"""
    WELL_DEFINED = "well_defined"  # Clear states and operators
    ILL_DEFINED = "ill_defined"  # Unclear states/operators

@dataclass
class ProblemState:
    """Represents a state in the problem space"""
    features: Dict  # Features describing the state
    salience_weights: Dict  # What aspects are salient
    operators: Set[str]  # Available actions
    constraints: Set[str]  # Path constraints

class ProblemFormulation:
    """Handles problem formulation and reframing based on relevance realization.
    
    This implements Vervaeke's framework for converting ill-defined problems
    into well-defined problems through dynamic restructuring of salience.
    """
    
    def __init__(self, relevance_core: RelevanceCore):
        self.relevance_core = relevance_core
        self.current_formulation: Optional[ProblemState] = None
        self.formulation_history: List[ProblemState] = []
        
    def formulate_problem(self, initial_state: Dict,
                         goal_state: Dict) -> ProblemState:
        """Create initial problem formulation.
        
        Args:
            initial_state: Features of starting state
            goal_state: Features of desired state
            
        Returns:
            Initial problem formulation
        """
        # Determine problem type
        problem_type = self._assess_problem_type(initial_state, goal_state)
        
        # Get relevant features using relevance core
        relevant_features = self.relevance_core.evaluate_relevance(
            set(initial_state.keys()) | set(goal_state.keys()),
            context={"problem_type": problem_type}
        )[0]
        
        # Create initial formulation
        formulation = ProblemState(
            features={
                k: v for k, v in initial_state.items()
                if k in relevant_features
            },
            salience_weights={
                k: 1.0 for k in relevant_features
            },
            operators=set(),  # To be determined through exploration
            constraints=set()  # To be determined through exploration
        )
        
        self.current_formulation = formulation
        self.formulation_history.append(formulation)
        
        return formulation
    
    def reframe_problem(self, new_context: Dict) -> ProblemState:
        """Reframe the problem based on new context/insight.
        
        This implements Vervaeke's insight mechanism for restructuring
        salience landscapes and problem representations.
        """
        if not self.current_formulation:
            raise ValueError("Must formulate problem before reframing")
            
        # Restructure relevance realization
        self.relevance_core.restructure_salience(
            RelevanceMode.PROBLEM_SPACE,
            new_context
        )
        
        # Get new relevant features
        relevant_features = self.relevance_core.evaluate_relevance(
            set(self.current_formulation.features.keys()),
            new_context
        )[0]
        
        # Create new formulation
        new_formulation = ProblemState(
            features={
                k: v for k, v in self.current_formulation.features.items()
                if k in relevant_features
            },
            salience_weights={
                k: new_context.get("salience_mod", 1.0)
                for k in relevant_features
            },
            operators=self.current_formulation.operators,
            constraints=self.current_formulation.constraints
        )
        
        self.current_formulation = new_formulation
        self.formulation_history.append(new_formulation)
        
        return new_formulation
    
    def _assess_problem_type(self, initial_state: Dict,
                           goal_state: Dict) -> ProblemType:
        """Determine if problem is well-defined or ill-defined."""
        # Check if states have clear features
        has_clear_states = (
            len(initial_state) > 0 and
            len(goal_state) > 0
        )
        
        # Check if operators can be determined
        can_determine_operators = self._can_determine_operators(
            initial_state, goal_state
        )
        
        if has_clear_states and can_determine_operators:
            return ProblemType.WELL_DEFINED
        return ProblemType.ILL_DEFINED
        
    def _can_determine_operators(self, initial_state: Dict,
                               goal_state: Dict) -> bool:
        """Check if operators can be clearly determined."""
        # Placeholder for more sophisticated operator analysis
        shared_keys = set(initial_state.keys()) & set(goal_state.keys())
        return len(shared_keys) > 0
        
    def evaluate_progress(self) -> float:
        """Evaluate progress in problem reformulation.
        
        Returns:
            Score between 0-1 indicating progress
        """
        if not self.formulation_history:
            return 0.0
            
        # Compare feature sets and salience weights
        initial = self.formulation_history[0]
        current = self.current_formulation
        
        # Check feature refinement
        feature_refinement = len(current.features) / len(initial.features)
        
        # Check salience restructuring 
        salience_change = np.mean([
            abs(current.salience_weights.get(k, 0) - 
                initial.salience_weights.get(k, 0))
            for k in set(current.features) | set(initial.features)
        ])
        
        # Combine metrics
        return (feature_refinement + salience_change) / 2.0 