"""
Core cognitive architecture implementing relevance realization mechanisms based on Vervaeke's framework.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class KnowingMode(Enum):
    PROPOSITIONAL = "propositional"  # Knowing that
    PROCEDURAL = "procedural"        # Knowing how
    PERSPECTIVAL = "perspectival"    # Knowing what it's like
    PARTICIPATORY = "participatory"   # Knowing by participating

@dataclass
class CognitiveFrame:
    """A cognitive frame represents a particular way of seeing/understanding"""
    salience_weights: Dict[str, float]  # What aspects are considered relevant
    active_knowing_modes: List[KnowingMode]
    context: Dict[str, Any]

class CognitiveCore:
    def __init__(self):
        self.frames: List[CognitiveFrame] = []
        self.current_frame: Optional[CognitiveFrame] = None
        self.salience_threshold = 0.5
        
    def realize_relevance(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Core relevance realization mechanism"""
        if not self.current_frame:
            return {}
            
        relevance_scores = {}
        for key, value in inputs.items():
            if key in self.current_frame.salience_weights:
                base_relevance = self.current_frame.salience_weights[key]
                # Modulate relevance based on context
                context_factor = self._compute_context_relevance(key, value)
                relevance_scores[key] = base_relevance * context_factor
                
        return {k:v for k,v in relevance_scores.items() 
                if v >= self.salience_threshold}
    
    def _compute_context_relevance(self, key: str, value: Any) -> float:
        """Compute contextual relevance factor"""
        if not self.current_frame:
            return 1.0
            
        context = self.current_frame.context
        if key not in context:
            return 1.0
            
        # Simple context matching for now
        if context[key] == value:
            return 2.0
        return 0.5
        
    def shift_frame(self, new_frame: CognitiveFrame):
        """Shift to a new cognitive frame"""
        self.frames.append(self.current_frame) if self.current_frame else None
        self.current_frame = new_frame
        
    def add_knowing_mode(self, mode: KnowingMode):
        """Add a new way of knowing to current frame"""
        if self.current_frame:
            self.current_frame.active_knowing_modes.append(mode)
            
    def get_active_knowing_modes(self) -> List[KnowingMode]:
        """Get currently active ways of knowing"""
        if not self.current_frame:
            return []
        return self.current_frame.active_knowing_modes 