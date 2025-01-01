from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict
from .agape_transformation import AgapeManager, LoveType
from .gnosis_transformation import GnosisManager, TransformationType

class TransformativeMode(Enum):
    PERSPECTIVAL = auto()  # Transformation of how we see
    PARTICIPATORY = auto()  # Transformation of how we engage
    ANAGOGIC = auto()  # Transformation through elevation
    METANOIC = auto()  # Radical reorientation of consciousness

@dataclass
class TransformativeState:
    """Represents the state of transformative experience."""
    mode: TransformativeMode
    is_egocentric: bool
    is_participatory: bool
    salience_landscape: Dict[str, float]
    worldview_coherence: float  # 0.0 to 1.0

class TransformativeDynamicsManager:
    """Manages the shared dynamics between Agape and Gnosis transformations."""
    
    def __init__(self):
        self.agape_manager = AgapeManager()
        self.gnosis_manager = GnosisManager()
        self.current_state = TransformativeState(
            mode=TransformativeMode.PERSPECTIVAL,
            is_egocentric=True,
            is_participatory=False,
            salience_landscape={"default": 1.0},
            worldview_coherence=0.0
        )
        
    def initiate_metanoia(self) -> bool:
        """Initiates radical reorientation through combined Agape and Gnosis."""
        # Start with Agapic transformation
        self.agape_manager.evolve_love_state(LoveType.AGAPE)
        
        # Integrate with Gnostic transformation
        gnosis_success = self.gnosis_manager.initiate_sensibility_transcendence(
            new_agent_perspective="transformed_self",
            new_arena_perspective="unified_world"
        )
        
        if gnosis_success:
            self.current_state = TransformativeState(
                mode=TransformativeMode.METANOIC,
                is_egocentric=False,
                is_participatory=True,
                salience_landscape={"meaning": 0.8, "connection": 0.9},
                worldview_coherence=0.7
            )
            return True
        return False
    
    def engage_participatory_knowing(self, context: str) -> bool:
        """Engages in participatory knowing through both love and insight."""
        if self.current_state.mode != TransformativeMode.METANOIC:
            return False
            
        # Apply Agapic forgiveness
        agape_success = self.agape_manager.apply_forgiveness()
        
        # Engage in Gnostic practice
        gnosis_success = self.gnosis_manager.engage_enactive_analogy(context)
        
        if agape_success and gnosis_success:
            self.current_state.worldview_coherence += 0.1
            self.current_state.salience_landscape["meaning"] += 0.1
            return True
        return False
    
    def assess_transformation(self) -> Dict[str, float]:
        """Assesses the current state of transformation."""
        agape_assessment = self.agape_manager.assess_transformation(
            self.agape_manager.transformations[-1] if self.agape_manager.transformations else None
        )
        gnosis_assessment = self.gnosis_manager.assess_gnosis_state()
        
        return {
            "coherence": self.current_state.worldview_coherence,
            "participation": float(self.current_state.is_participatory),
            "egocentrism": float(self.current_state.is_egocentric),
            "agape_active": float(agape_assessment if agape_assessment is not None else 0.0),
            "gnosis_active": float(not gnosis_assessment["is_existentially_trapped"])
        }
    
    def facilitate_anagoge(self, practices: List[str]) -> bool:
        """Facilitates anagogic ascent through combined practices."""
        if self.current_state.mode not in [TransformativeMode.METANOIC, TransformativeMode.PARTICIPATORY]:
            return False
            
        # Apply psychotechnologies through Gnosis
        gnosis_success = self.gnosis_manager.integrate_psychotechnologies(practices)
        
        # Evolve love state if Gnosis is successful
        if gnosis_success:
            self.agape_manager.evolve_love_state(LoveType.AGAPE)
            self.current_state.mode = TransformativeMode.ANAGOGIC
            self.current_state.worldview_coherence += 0.2
            return True
        return False 