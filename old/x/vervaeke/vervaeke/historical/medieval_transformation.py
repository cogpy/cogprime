from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict

class EpistemicMode(Enum):
    PARTICIPATORY = auto()  # Direct engagement with reality
    REPRESENTATIONAL = auto()  # Reality mediated through mental representations
    NOMINALISTIC = auto()  # Reality constructed through language

class RealityDomain(Enum):
    NATURAL = auto()  # Physical, observable world
    SUPERNATURAL = auto()  # Transcendent, divine realm
    EXPERIENTIAL = auto()  # Domain of lived experience

@dataclass
class EpistemicState:
    """Represents the state of knowing and meaning-making."""
    mode: EpistemicMode
    reality_access: Dict[RealityDomain, float]  # 0-1 scale of access to each domain
    coherence_level: float  # 0-1 scale of internal consistency
    experiential_grounding: float  # 0-1 scale of connection to lived experience

class MedievalTransformation:
    """Models the transformation of meaning during the medieval period."""
    
    def __init__(self):
        self.epistemic_state = EpistemicState(
            mode=EpistemicMode.PARTICIPATORY,
            reality_access={
                RealityDomain.NATURAL: 1.0,
                RealityDomain.SUPERNATURAL: 1.0,
                RealityDomain.EXPERIENTIAL: 1.0
            },
            coherence_level=1.0,
            experiential_grounding=1.0
        )
        self.transformations: List[str] = []

    def apply_aquinas_division(self) -> None:
        """Models Aquinas's division between natural and supernatural realms."""
        self.epistemic_state.mode = EpistemicMode.REPRESENTATIONAL
        self.epistemic_state.reality_access[RealityDomain.SUPERNATURAL] *= 0.5
        self.epistemic_state.coherence_level *= 0.8
        self.transformations.append(
            "Aquinas divides reality into natural and supernatural domains"
        )

    def apply_nominalistic_shift(self) -> None:
        """Models the shift towards nominalism and linguistic construction."""
        self.epistemic_state.mode = EpistemicMode.NOMINALISTIC
        self.epistemic_state.reality_access[RealityDomain.NATURAL] *= 0.7
        self.epistemic_state.reality_access[RealityDomain.SUPERNATURAL] *= 0.3
        self.epistemic_state.experiential_grounding *= 0.5
        self.transformations.append(
            "Shift to nominalistic understanding of reality through language"
        )

@dataclass
class WillDynamics:
    """Models the dynamics of will in medieval thought."""
    self_assertion: float  # 0-1 scale
    divine_receptivity: float  # 0-1 scale
    inner_conflict: float  # 0-1 scale

class RhinelandMysticism:
    """Models the transformation of spirituality in Rhineland mysticism."""
    
    def __init__(self):
        self.will_dynamics = WillDynamics(
            self_assertion=0.5,
            divine_receptivity=0.5,
            inner_conflict=0.0
        )
        self.transformations: List[str] = []

    def apply_will_negation(self, intensity: float) -> None:
        """Models the practice of will negation."""
        self.will_dynamics.self_assertion *= (1 - intensity)
        self.will_dynamics.divine_receptivity += (intensity * 0.5)
        self.will_dynamics.inner_conflict += (intensity * 0.3)
        self.transformations.append(
            f"Will negation applied with intensity {intensity:.2f}"
        )

    def assess_spiritual_state(self) -> List[str]:
        """Assesses the current spiritual state."""
        assessment = []
        if self.will_dynamics.self_assertion < 0.3:
            assessment.append("Strong self-negation present")
        if self.will_dynamics.divine_receptivity > 0.7:
            assessment.append("High divine receptivity achieved")
        if self.will_dynamics.inner_conflict > 0.6:
            assessment.append("Significant inner conflict manifesting")
        return assessment

class MeaningTransformationTracker:
    """Tracks the overall transformation of meaning through medieval period."""
    
    def __init__(self):
        self.medieval_transformation = MedievalTransformation()
        self.rhineland_mysticism = RhinelandMysticism()
        self.historical_stages: List[str] = []

    def advance_historical_stage(self, description: str) -> None:
        """Records a major historical stage in meaning transformation."""
        self.historical_stages.append(description)
        
    def assess_meaning_viability(self) -> Dict[str, float]:
        """Assesses the viability of different aspects of meaning."""
        return {
            "participatory_knowing": 
                1.0 if self.medieval_transformation.epistemic_state.mode == EpistemicMode.PARTICIPATORY else 0.3,
            "reality_access": 
                sum(self.medieval_transformation.epistemic_state.reality_access.values()) / 3,
            "spiritual_integration":
                (self.rhineland_mysticism.will_dynamics.divine_receptivity - 
                 self.rhineland_mysticism.will_dynamics.inner_conflict),
            "overall_coherence":
                self.medieval_transformation.epistemic_state.coherence_level
        } 