from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

class ExperienceMode(Enum):
    """Modes of experiencing based on Vervaeke's framework"""
    HAVING = "having"  # Focal, problem-solving mode
    BEING = "being"   # Non-focal, participatory mode

class PhenomenologicalState(Enum):
    """States of phenomenological experience"""
    WONDER = "wonder"   # Opening up to gestalt
    AWE = "awe"        # Accommodation and transformation
    MYSTERY = "mystery" # Non-focal awareness of framing
    HORROR = "horror"  # Overwhelming combinatorial explosion

@dataclass
class TransformativeExperience:
    """Represents a transformative experience through wonder/awe"""
    initial_frame: Dict  # Initial cognitive frame
    transformation: str  # Nature of the transformation
    new_frame: Dict     # Emergent cognitive frame
    affective_tone: float  # Emotional valence (-1 to 1)
    
class PhenomenologyCore:
    """Handles phenomenological aspects of relevance realization.
    
    Implements Vervaeke's account of wonder, awe, and mystery as ways
    of experiencing relevance realization non-focally.
    """
    
    def __init__(self):
        self.current_mode = ExperienceMode.HAVING
        self.current_state = None
        self.frame_history: List[Dict] = []
        self.mystery_threshold = 0.8
        
    def shift_mode(self, 
                  context: Dict,
                  force_shift: bool = False) -> ExperienceMode:
        """Shift between Having and Being modes based on context.
        
        Args:
            context: Current cognitive context
            force_shift: Whether to force mode shift
            
        Returns:
            New experience mode
        """
        # Compute indicators for mode shift
        novelty = self._compute_novelty(context)
        complexity = self._compute_complexity(context)
        
        # Shift to Being mode if:
        # - High novelty (something surprising/wonderful)
        # - High complexity (beyond focal processing)
        # - Explicitly requested
        if force_shift or novelty > 0.7 or complexity > 0.8:
            self.current_mode = ExperienceMode.BEING
        else:
            self.current_mode = ExperienceMode.HAVING
            
        return self.current_mode
        
    def induce_wonder(self,
                     context: Dict,
                     intensity: float = 0.5) -> Optional[TransformativeExperience]:
        """Attempt to induce wonder through opening to gestalt.
        
        Args:
            context: Current cognitive context
            intensity: Desired intensity of wonder (0-1)
            
        Returns:
            Transformative experience if successful
        """
        # Shift to Being mode
        self.shift_mode(context, force_shift=True)
        
        # Attempt gestalt shift
        if self._can_shift_gestalt(context):
            # Record initial frame
            initial = context.copy()
            
            # Open up frame through wonder
            new_frame = self._open_frame(context, intensity)
            
            # Create transformative experience
            experience = TransformativeExperience(
                initial_frame=initial,
                transformation="wonder_opening",
                new_frame=new_frame,
                affective_tone=0.7 * intensity
            )
            
            self.current_state = PhenomenologicalState.WONDER
            return experience
            
        return None
        
    def induce_awe(self,
                   context: Dict,
                   intensity: float = 0.5) -> Optional[TransformativeExperience]:
        """Attempt to induce awe through accommodation.
        
        Args:
            context: Current cognitive context
            intensity: Desired intensity of awe (0-1)
            
        Returns:
            Transformative experience if successful
        """
        # Shift to Being mode
        self.shift_mode(context, force_shift=True)
        
        # Check if context warrants awe
        if self._warrants_awe(context):
            # Record initial frame
            initial = context.copy()
            
            # Transform frame through accommodation
            new_frame = self._accommodate_frame(context, intensity)
            
            # Create transformative experience
            experience = TransformativeExperience(
                initial_frame=initial,
                transformation="awe_accommodation", 
                new_frame=new_frame,
                affective_tone=0.9 * intensity
            )
            
            self.current_state = PhenomenologicalState.AWE
            return experience
            
        return None
        
    def engage_mystery(self,
                      context: Dict) -> Tuple[PhenomenologicalState, float]:
        """Engage with mystery through non-focal awareness.
        
        Args:
            context: Current cognitive context
            
        Returns:
            Tuple of (phenomenological state, mystery intensity)
        """
        # Compute mystery indicators
        frame_recursion = self._compute_frame_recursion(context)
        ineffability = self._compute_ineffability(context)
        
        # Mystery emerges when:
        # - High frame recursion (frames within frames)
        # - High ineffability (resistance to focal grasp)
        mystery_intensity = (frame_recursion + ineffability) / 2
        
        if mystery_intensity > self.mystery_threshold:
            self.current_state = PhenomenologicalState.MYSTERY
        elif mystery_intensity < 0.2:
            self.current_state = PhenomenologicalState.HORROR
        
        return self.current_state, mystery_intensity
        
    def _compute_novelty(self, context: Dict) -> float:
        """Compute novelty of current context."""
        if not self.frame_history:
            return 1.0
            
        # Compare with recent frames
        similarities = [
            self._frame_similarity(context, past)
            for past in self.frame_history[-3:]
        ]
        return 1 - np.mean(similarities)
        
    def _compute_complexity(self, context: Dict) -> float:
        """Compute cognitive complexity of context."""
        # Analyze frame structure complexity
        n_elements = len(context.get("elements", []))
        n_relations = len(context.get("relations", []))
        
        complexity = np.tanh(0.1 * (n_elements + 2 * n_relations))
        return complexity
        
    def _can_shift_gestalt(self, context: Dict) -> bool:
        """Check if context allows for gestalt shift."""
        # Need sufficient elements but not overwhelming
        n_elements = len(context.get("elements", []))
        return 3 <= n_elements <= 12
        
    def _open_frame(self, context: Dict, intensity: float) -> Dict:
        """Open up cognitive frame through wonder."""
        new_frame = context.copy()
        
        # Expand frame boundaries
        new_frame["boundaries"] = [
            b * (1 + 0.5 * intensity)
            for b in context.get("boundaries", [1, 1])
        ]
        
        # Increase element connectivity
        new_frame["relations"] = context.get("relations", []) + [
            {"type": "potential", "strength": 0.5 * intensity}
            for _ in range(int(3 * intensity))
        ]
        
        return new_frame
        
    def _warrants_awe(self, context: Dict) -> bool:
        """Check if context warrants awe response."""
        # Need high complexity and low familiarity
        complexity = self._compute_complexity(context)
        novelty = self._compute_novelty(context)
        
        return complexity > 0.7 and novelty > 0.6
        
    def _accommodate_frame(self, context: Dict, intensity: float) -> Dict:
        """Transform frame through accommodation."""
        new_frame = context.copy()
        
        # Fundamental restructuring
        new_frame["structure"] = "emergent"
        new_frame["boundaries"] = None  # Dissolve boundaries
        
        # Add transformative elements
        new_frame["transformations"] = context.get("transformations", []) + [{
            "type": "accommodation",
            "intensity": intensity,
            "timestamp": "now"
        }]
        
        return new_frame
        
    def _compute_frame_recursion(self, context: Dict) -> float:
        """Compute degree of frame recursion."""
        depth = 0
        current = context
        
        # Track recursive frame embedding
        while "meta_frame" in current:
            depth += 1
            current = current["meta_frame"]
            
        return np.tanh(0.5 * depth)
        
    def _compute_ineffability(self, context: Dict) -> float:
        """Compute resistance to focal grasp."""
        # Indicators of ineffability:
        # - Lack of clear boundaries
        # - High interconnectivity
        # - Presence of paradox
        
        has_boundaries = "boundaries" in context
        n_relations = len(context.get("relations", []))
        has_paradox = any(
            r.get("type") == "paradox"
            for r in context.get("relations", [])
        )
        
        ineffability = (
            (0.4 if not has_boundaries else 0) +
            (0.3 * np.tanh(0.1 * n_relations)) +
            (0.3 if has_paradox else 0)
        )
        
        return ineffability
        
    def _frame_similarity(self, frame1: Dict, frame2: Dict) -> float:
        """Compute similarity between two cognitive frames."""
        # Compare frame elements
        elements1 = set(str(e) for e in frame1.get("elements", []))
        elements2 = set(str(e) for e in frame2.get("elements", []))
        
        if not elements1 or not elements2:
            return 0.0
            
        # Jaccard similarity
        similarity = len(elements1 & elements2) / len(elements1 | elements2)
        return similarity 