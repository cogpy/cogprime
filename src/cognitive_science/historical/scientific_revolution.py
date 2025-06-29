from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional

class RealityProperty(Enum):
    OBJECTIVE = auto()  # Mathematically measurable properties
    SUBJECTIVE = auto()  # Experiential, value-laden properties
    PURPOSIVE = auto()  # Goal-directed, meaningful properties

class MatterConception(Enum):
    POTENTIAL = auto()  # Matter as potential for information
    INERT = auto()     # Matter as resistant, lifeless substance
    MECHANICAL = auto() # Matter as part of universal machine

@dataclass
class GalileanTransformation:
    """Models Galileo's transformation of reality conception."""
    mathematical_abstraction: float  # 0-1 scale of mathematical reduction
    experiential_disconnection: float  # 0-1 scale of separation from experience
    mechanistic_view: float  # 0-1 scale of mechanical worldview

@dataclass
class PropertyDomain:
    """Represents the domain of properties in reality."""
    reality_properties: Dict[RealityProperty, float]  # 0-1 scale of prominence
    matter_conception: MatterConception
    universe_animation: float  # 0-1 scale (1 = fully alive, 0 = fully dead)

class ScientificRevolution:
    """Models the transformation brought by the scientific revolution."""
    
    def __init__(self):
        self.property_domain = PropertyDomain(
            reality_properties={
                RealityProperty.OBJECTIVE: 0.3,
                RealityProperty.SUBJECTIVE: 0.7,
                RealityProperty.PURPOSIVE: 0.8
            },
            matter_conception=MatterConception.POTENTIAL,
            universe_animation=1.0
        )
        self.galilean_state = GalileanTransformation(
            mathematical_abstraction=0.0,
            experiential_disconnection=0.0,
            mechanistic_view=0.0
        )
        self.transformations: List[str] = []

    def apply_galilean_revolution(self) -> None:
        """Applies Galileo's revolutionary changes to reality conception."""
        # Shift property emphasis
        self.property_domain.reality_properties[RealityProperty.OBJECTIVE] = 0.9
        self.property_domain.reality_properties[RealityProperty.SUBJECTIVE] = 0.2
        self.property_domain.reality_properties[RealityProperty.PURPOSIVE] = 0.1
        
        # Transform matter conception
        self.property_domain.matter_conception = MatterConception.MECHANICAL
        self.property_domain.universe_animation = 0.1
        
        # Update Galilean state
        self.galilean_state.mathematical_abstraction = 0.9
        self.galilean_state.experiential_disconnection = 0.8
        self.galilean_state.mechanistic_view = 0.9
        
        self.transformations.append("Galilean revolution applied: Reality mathematized")

@dataclass
class ExperientialImpact:
    """Models the impact on human experience of reality."""
    world_connection: float  # 0-1 scale
    meaning_viability: float  # 0-1 scale
    agency_sense: float  # 0-1 scale
    purpose_sense: float  # 0-1 scale

class MeaningCrisisProgression:
    """Tracks the progression of the meaning crisis through scientific revolution."""
    
    def __init__(self):
        self.scientific_revolution = ScientificRevolution()
        self.experiential_impact = ExperientialImpact(
            world_connection=1.0,
            meaning_viability=1.0,
            agency_sense=1.0,
            purpose_sense=1.0
        )
        self.stages: List[str] = []

    def apply_scientific_transformation(self) -> None:
        """Applies scientific revolution's transformation and tracks impact."""
        # Apply Galilean changes
        self.scientific_revolution.apply_galilean_revolution()
        
        # Update experiential impact
        self.experiential_impact.world_connection *= 0.3
        self.experiential_impact.meaning_viability *= 0.4
        self.experiential_impact.agency_sense *= 0.5
        self.experiential_impact.purpose_sense *= 0.2
        
        self.stages.append(
            "Scientific revolution transforms reality conception and human experience"
        )

    def assess_crisis_state(self) -> Dict[str, float]:
        """Assesses the current state of the meaning crisis."""
        return {
            "reality_coherence": 
                sum(self.scientific_revolution.property_domain.reality_properties.values()) / 3,
            "experiential_integrity":
                (self.experiential_impact.world_connection + 
                 self.experiential_impact.meaning_viability) / 2,
            "human_situation":
                (self.experiential_impact.agency_sense + 
                 self.experiential_impact.purpose_sense) / 2,
            "overall_crisis_severity":
                1.0 - ((self.experiential_impact.world_connection + 
                        self.experiential_impact.meaning_viability + 
                        self.experiential_impact.agency_sense + 
                        self.experiential_impact.purpose_sense) / 4)
        } 