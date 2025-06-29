"""
Module for handling 4E cognitive science components based on Vervaeke's framework.
Implements mechanisms for embodied, embedded, enactive, and extended cognition.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

class CognitiveAspect(Enum):
    EMBODIED = "embodied"
    EMBEDDED = "embedded"
    ENACTIVE = "enactive"
    EXTENDED = "extended"
    EMERGENT = "emergent"

class EmotionalState(Enum):
    CARING = "caring"
    COPING = "coping"
    AGAPE = "agape"
    WONDER = "wonder"
    FLOW = "flow"

@dataclass
class CognitiveContext:
    """Represents the context in which cognition occurs."""
    body_state: Dict[str, float]  # Physiological measures
    environment: Dict[str, object]  # Environmental factors
    social_context: Dict[str, object]  # Social relationships
    tools: Set[str]  # Extended cognitive tools
    emotional_state: EmotionalState

@dataclass
class EmergentProperty:
    """Represents an emergent property from cognitive processes."""
    name: str
    source_aspects: Set[CognitiveAspect]
    conditions: Dict[str, float]
    effects: Dict[str, float]

class CognitiveCore:
    """Core class for managing 4E cognitive processes."""
    
    def __init__(self):
        self.active_aspects: Set[CognitiveAspect] = set()
        self.context: Optional[CognitiveContext] = None
        self.emergent_properties: List[EmergentProperty] = []
        
    def update_context(self, context: CognitiveContext) -> None:
        """Update the cognitive context."""
        self.context = context
        self._adapt_to_context()
        
    def _adapt_to_context(self) -> None:
        """Adapt cognitive processes to current context."""
        if not self.context:
            return
            
        # Adjust active aspects based on context
        if self.context.body_state:
            self.active_aspects.add(CognitiveAspect.EMBODIED)
        if self.context.environment:
            self.active_aspects.add(CognitiveAspect.EMBEDDED)
        if self.context.social_context:
            self.active_aspects.add(CognitiveAspect.ENACTIVE)
        if self.context.tools:
            self.active_aspects.add(CognitiveAspect.EXTENDED)
            
    def process_emergence(self) -> List[EmergentProperty]:
        """Process emergent properties from current cognitive state."""
        if not self.context:
            return []
            
        new_properties = []
        # Check for emergence conditions
        if CognitiveAspect.EMBODIED in self.active_aspects and \
           CognitiveAspect.EMBEDDED in self.active_aspects:
            new_properties.append(
                EmergentProperty(
                    name="embodied_coupling",
                    source_aspects={CognitiveAspect.EMBODIED, CognitiveAspect.EMBEDDED},
                    conditions={"coupling_strength": 0.8},
                    effects={"adaptive_capacity": 0.9}
                )
            )
            
        if len(self.active_aspects) >= 3:
            new_properties.append(
                EmergentProperty(
                    name="cognitive_integration",
                    source_aspects=self.active_aspects,
                    conditions={"integration_level": 0.7},
                    effects={"meaning_making": 0.8}
                )
            )
            
        self.emergent_properties.extend(new_properties)
        return new_properties
        
    def cultivate_agape(self) -> float:
        """Cultivate agapic love through cognitive integration."""
        if not self.context or self.context.emotional_state != EmotionalState.CARING:
            return 0.0
            
        active_aspects_count = len(self.active_aspects)
        emergent_properties_count = len(self.emergent_properties)
        
        # Agape emerges from integration of cognitive aspects and emotional caring
        agape_strength = (
            (active_aspects_count / len(CognitiveAspect)) * 0.5 +
            (emergent_properties_count / (active_aspects_count * 2)) * 0.5
        )
        
        return min(1.0, agape_strength)
        
    def evaluate_cognitive_excellence(self) -> Dict[str, float]:
        """Evaluate cognitive excellence across different dimensions."""
        if not self.context:
            return {}
            
        return {
            "embodied_wisdom": len([p for p in self.emergent_properties 
                                  if CognitiveAspect.EMBODIED in p.source_aspects]) / 10.0,
            "social_integration": len([p for p in self.emergent_properties 
                                     if CognitiveAspect.ENACTIVE in p.source_aspects]) / 10.0,
            "tool_mastery": len([p for p in self.emergent_properties 
                               if CognitiveAspect.EXTENDED in p.source_aspects]) / 10.0,
            "adaptive_capacity": len(self.emergent_properties) / 20.0
        } 