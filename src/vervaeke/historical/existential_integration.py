from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict
from .agape_transformation import AgapeManager, LoveType
from .gnosis_transformation import GnosisManager, WorldviewState

class ExistentialMode(Enum):
    ENTRAPPED = auto()  # Stuck in existential inertia
    AWAKENING = auto()  # Beginning to see beyond entrapment
    LIBERATING = auto()  # Active process of liberation
    TRANSCENDENT = auto()  # Beyond existential entrapment

@dataclass
class ExistentialState:
    """Represents the state of existential being."""
    mode: ExistentialMode
    meaning_coherence: float  # 0.0 to 1.0
    self_transcendence: float  # 0.0 to 1.0
    worldview_viability: Dict[str, float]
    is_authentically_engaged: bool

class ExistentialIntegrationManager:
    """Manages the integration of Agape and Gnosis in existential transformation."""
    
    def __init__(self):
        self.agape_manager = AgapeManager()
        self.gnosis_manager = GnosisManager()
        self.current_state = ExistentialState(
            mode=ExistentialMode.ENTRAPPED,
            meaning_coherence=0.0,
            self_transcendence=0.0,
            worldview_viability={"default": 0.5},
            is_authentically_engaged=False
        )
        
    def initiate_awakening(self) -> bool:
        """Initiates existential awakening through combined Agape and Gnosis."""
        # Begin with Agapic transformation
        self.agape_manager.evolve_love_state(LoveType.AGAPE)
        
        # Integrate with Gnostic awakening
        gnosis_success = self.gnosis_manager.initiate_sensibility_transcendence(
            new_agent_perspective="awakened_self",
            new_arena_perspective="meaningful_world"
        )
        
        if gnosis_success:
            self.current_state = ExistentialState(
                mode=ExistentialMode.AWAKENING,
                meaning_coherence=0.6,
                self_transcendence=0.4,
                worldview_viability={"meaning": 0.7, "connection": 0.8},
                is_authentically_engaged=True
            )
            return True
        return False
    
    def deepen_engagement(self, context: str) -> bool:
        """Deepens existential engagement through love and insight."""
        if self.current_state.mode == ExistentialMode.ENTRAPPED:
            return False
            
        # Apply Agapic forgiveness
        agape_success = self.agape_manager.apply_forgiveness()
        
        # Engage in Gnostic practice
        gnosis_success = self.gnosis_manager.engage_enactive_analogy(context)
        
        if agape_success and gnosis_success:
            self.current_state.meaning_coherence += 0.1
            self.current_state.self_transcendence += 0.1
            if self.current_state.mode == ExistentialMode.AWAKENING:
                self.current_state.mode = ExistentialMode.LIBERATING
            return True
        return False
    
    def assess_integration(self) -> Dict[str, float]:
        """Assesses the current state of existential integration."""
        agape_assessment = self.agape_manager.assess_transformation(
            self.agape_manager.transformations[-1] if self.agape_manager.transformations else None
        )
        gnosis_assessment = self.gnosis_manager.assess_gnosis_state()
        
        return {
            "meaning_coherence": self.current_state.meaning_coherence,
            "self_transcendence": self.current_state.self_transcendence,
            "authentic_engagement": float(self.current_state.is_authentically_engaged),
            "agape_active": float(agape_assessment if agape_assessment is not None else 0.0),
            "gnosis_active": float(not gnosis_assessment["is_existentially_trapped"])
        }
    
    def facilitate_transcendence(self, practices: List[str]) -> bool:
        """Facilitates existential transcendence through combined practices."""
        if self.current_state.mode not in [ExistentialMode.AWAKENING, ExistentialMode.LIBERATING]:
            return False
            
        # Apply psychotechnologies through Gnosis
        gnosis_success = self.gnosis_manager.integrate_psychotechnologies(practices)
        
        # Evolve love state if Gnosis is successful
        if gnosis_success:
            self.agape_manager.evolve_love_state(LoveType.AGAPE)
            if self.current_state.mode == ExistentialMode.LIBERATING:
                self.current_state.mode = ExistentialMode.TRANSCENDENT
                self.current_state.meaning_coherence += 0.2
                self.current_state.self_transcendence += 0.2
            return True
        return False 