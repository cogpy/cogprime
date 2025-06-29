from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict
from .agape_transformation import AgapeManager, LoveType, TransformativeState
from .gnosis_transformation import GnosisManager, WorldviewState, TransformationType

class IntegrationType(Enum):
    LOVE_BASED = auto()  # Transformation through Agapic love
    GNOSIS_BASED = auto()  # Transformation through Gnostic insight
    COMBINED = auto()  # Synergistic transformation through both

@dataclass
class IntegratedTransformation:
    """Represents a transformation that combines Agape and Gnosis."""
    love_component: TransformativeState
    gnosis_component: WorldviewState
    integration_type: IntegrationType
    is_complete: bool
    synergy_level: float  # 0.0 to 1.0 measure of integration

class TransformativeIntegrationManager:
    """Manages the integration of Agapic and Gnostic transformation."""
    
    def __init__(self):
        self.agape_manager = AgapeManager()
        self.gnosis_manager = GnosisManager()
        self.integrated_transformations: List[IntegratedTransformation] = []
        
    def initiate_integrated_transformation(self) -> IntegratedTransformation:
        """Initiates a transformation that combines Agape and Gnosis."""
        # Start with Agapic transformation
        self.agape_manager.evolve_love_state(LoveType.AGAPE)
        
        # Initialize corresponding Gnostic transformation
        self.gnosis_manager.initiate_sensibility_transcendence(
            new_agent_perspective="agapic_self",
            new_arena_perspective="interconnected_world"
        )
        
        # Create integrated transformation
        transformation = IntegratedTransformation(
            love_component=self.agape_manager.current_state,
            gnosis_component=self.gnosis_manager.current_state.current_worldview,
            integration_type=IntegrationType.COMBINED,
            is_complete=False,
            synergy_level=0.0
        )
        
        self.integrated_transformations.append(transformation)
        return transformation
    
    def apply_agapic_gnosis(self, context: str) -> bool:
        """Applies both Agapic love and Gnostic insight to transformation."""
        if not self.integrated_transformations:
            return False
            
        current_transformation = self.integrated_transformations[-1]
        
        # Apply forgiveness through Agape
        agape_success = self.agape_manager.apply_forgiveness()
        
        # Engage in Gnostic practice
        gnosis_success = self.gnosis_manager.engage_enactive_analogy(context)
        
        if agape_success and gnosis_success:
            current_transformation.synergy_level += 0.2
            if current_transformation.synergy_level >= 1.0:
                current_transformation.is_complete = True
            return True
        return False
    
    def assess_integration(self) -> Dict[str, float]:
        """Assesses the current state of integration between Agape and Gnosis."""
        if not self.integrated_transformations:
            return {"integration_level": 0.0}
            
        current_transformation = self.integrated_transformations[-1]
        
        # Assess both aspects
        agape_assessment = self.agape_manager.assess_transformation(
            self.agape_manager.transformations[-1] if self.agape_manager.transformations else None
        )
        gnosis_assessment = self.gnosis_manager.assess_gnosis_state()
        
        # Calculate integration metrics
        return {
            "integration_level": current_transformation.synergy_level,
            "agape_active": float(agape_assessment if agape_assessment is not None else 0.0),
            "gnosis_active": float(not gnosis_assessment["is_existentially_trapped"]),
            "transformation_complete": float(current_transformation.is_complete)
        }
    
    def facilitate_transformation(self, techniques: List[str]) -> bool:
        """Facilitates transformation using both Agapic and Gnostic methods."""
        if not self.integrated_transformations:
            self.initiate_integrated_transformation()
            
        # Apply psychotechnologies through Gnosis
        gnosis_success = self.gnosis_manager.integrate_psychotechnologies(techniques)
        
        # Evolve love state if Gnosis is successful
        if gnosis_success:
            self.agape_manager.evolve_love_state(LoveType.AGAPE)
            current_transformation = self.integrated_transformations[-1]
            current_transformation.synergy_level += 0.3
            if current_transformation.synergy_level >= 1.0:
                current_transformation.is_complete = True
            return True
        return False 