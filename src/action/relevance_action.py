"""
Action module for generating actions based on relevance realization.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from ..core.vervaeke_cognitive_core import CognitiveFrame, KnowingMode

class ActionType(Enum):
    ATTEND = "attend"           # Shift attention
    FRAME_SHIFT = "frame_shift" # Change cognitive frame
    EXPLORE = "explore"         # Gather new information
    EXPLOIT = "exploit"         # Use existing knowledge
    REFLECT = "reflect"         # Meta-cognitive action

@dataclass
class Action:
    """Represents an action the system can take"""
    type: ActionType
    target: Dict[str, Any]  # What to act on
    knowing_modes: List[KnowingMode]  # Ways of knowing involved
    expected_relevance: float
    confidence: float

class ActionGenerator:
    """Generates actions based on relevance realization"""
    
    def __init__(self, relevance_threshold: float = 0.7):
        self.relevance_threshold = relevance_threshold
        
    def generate_actions(self, 
                        current_frame: CognitiveFrame,
                        relevance_scores: Dict[str, float],
                        context: Dict[str, Any]) -> List[Action]:
        """Generate possible actions based on current relevance realization"""
        actions = []
        
        # Generate attention actions for highly relevant items
        for item, relevance in relevance_scores.items():
            if relevance >= self.relevance_threshold:
                actions.append(Action(
                    type=ActionType.ATTEND,
                    target={'item': item},
                    knowing_modes=[KnowingMode.PERSPECTIVAL],
                    expected_relevance=relevance,
                    confidence=min(1.0, relevance)
                ))
                
        # Generate frame shift actions if context suggests it
        if self._should_shift_frame(current_frame, context):
            actions.append(Action(
                type=ActionType.FRAME_SHIFT,
                target={'context': context},
                knowing_modes=[KnowingMode.PARTICIPATORY],
                expected_relevance=self._compute_frame_shift_relevance(context),
                confidence=0.8
            ))
            
        # Generate exploration actions for low-confidence areas
        if self._should_explore(relevance_scores):
            actions.append(Action(
                type=ActionType.EXPLORE,
                target={'areas': self._identify_exploration_areas(relevance_scores)},
                knowing_modes=[KnowingMode.PROCEDURAL],
                expected_relevance=0.5,  # Moderate relevance for exploration
                confidence=0.6
            ))
            
        # Generate exploitation actions for high-confidence areas
        if self._should_exploit(relevance_scores):
            actions.append(Action(
                type=ActionType.EXPLOIT,
                target={'items': self._identify_exploitation_targets(relevance_scores)},
                knowing_modes=[KnowingMode.PROPOSITIONAL, KnowingMode.PROCEDURAL],
                expected_relevance=0.9,
                confidence=0.9
            ))
            
        return actions
        
    def select_best_action(self, actions: List[Action]) -> Optional[Action]:
        """Select the best action based on relevance and confidence"""
        if not actions:
            return None
            
        # Compute combined score considering both relevance and confidence
        def action_score(action: Action) -> float:
            return 0.7 * action.expected_relevance + 0.3 * action.confidence
            
        return max(actions, key=action_score)
        
    def _should_shift_frame(self, frame: CognitiveFrame, 
                           context: Dict[str, Any]) -> bool:
        """Determine if frame shift is needed"""
        # Simple heuristic: shift if context keys differ significantly
        context_overlap = len(set(frame.context.keys()) & set(context.keys()))
        return context_overlap / max(len(frame.context), len(context)) < 0.5
        
    def _compute_frame_shift_relevance(self, context: Dict[str, Any]) -> float:
        """Compute relevance of frame shift"""
        # Simple heuristic based on context size
        return min(1.0, len(context) / 10)
        
    def _should_explore(self, relevance_scores: Dict[str, float]) -> bool:
        """Determine if exploration is needed"""
        # Explore if average relevance is low
        if not relevance_scores:
            return True
        return sum(relevance_scores.values()) / len(relevance_scores) < 0.4
        
    def _should_exploit(self, relevance_scores: Dict[str, float]) -> bool:
        """Determine if exploitation is warranted"""
        # Exploit if we have high relevance items
        return any(score > 0.8 for score in relevance_scores.values())
        
    def _identify_exploration_areas(self, 
                                  relevance_scores: Dict[str, float]) -> List[str]:
        """Identify areas needing exploration"""
        return [item for item, score in relevance_scores.items() 
                if score < 0.3]
        
    def _identify_exploitation_targets(self,
                                     relevance_scores: Dict[str, float]) -> List[str]:
        """Identify targets for exploitation"""
        return [item for item, score in relevance_scores.items() 
                if score > 0.8] 