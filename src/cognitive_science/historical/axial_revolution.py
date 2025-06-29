from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

class AxialTradition(Enum):
    """Major traditions emerging from the Axial Revolution"""
    GREEK = "greek"          # Greek philosophical tradition
    JUDAIC = "judaic"       # Judaic prophetic tradition
    BUDDHIST = "buddhist"   # Buddhist contemplative tradition
    CONFUCIAN = "confucian" # Confucian wisdom tradition

class AxialInnovation(Enum):
    """Key innovations of the Axial Revolution"""
    SECOND_ORDER_THINKING = "second_order_thinking"
    SYSTEMATIC_RATIONALITY = "systematic_rationality"
    TRANSCENDENT_CRITIQUE = "transcendent_critique"
    PARTICIPATORY_KNOWING = "participatory_knowing"
    TRANSFORMATIVE_PRACTICES = "transformative_practices"

@dataclass
class AxialState:
    """State of Axial development and transformation"""
    active_traditions: Dict[AxialTradition, float]
    innovations: Dict[AxialInnovation, float]
    integration_level: float
    transformation_depth: float

class AxialRevolution:
    """Manages the transformative dynamics of the Axial Revolution"""
    
    def __init__(self):
        self.state = AxialState(
            active_traditions={tradition: 0.0 for tradition in AxialTradition},
            innovations={innovation: 0.0 for innovation in AxialInnovation},
            integration_level=0.0,
            transformation_depth=0.0
        )
        
    def activate_tradition(
        self,
        tradition: AxialTradition,
        intensity: float = 0.5
    ) -> None:
        """Activate an Axial tradition with specified intensity"""
        self.state.active_traditions[tradition] = min(
            1.0,
            self.state.active_traditions[tradition] + intensity
        )
        self._update_innovations()
        self._update_integration()
        
    def enhance_innovation(
        self,
        innovation: AxialInnovation,
        intensity: float = 0.5
    ) -> None:
        """Enhance an Axial innovation with specified intensity"""
        self.state.innovations[innovation] = min(
            1.0,
            self.state.innovations[innovation] + intensity
        )
        self._update_integration()
        self._update_transformation()
        
    def _update_innovations(self) -> None:
        """Update innovations based on active traditions"""
        # Greek tradition enhances systematic rationality
        if self.state.active_traditions[AxialTradition.GREEK] > 0.5:
            self.state.innovations[AxialInnovation.SYSTEMATIC_RATIONALITY] += 0.2
            
        # Judaic tradition enhances transcendent critique
        if self.state.active_traditions[AxialTradition.JUDAIC] > 0.5:
            self.state.innovations[AxialInnovation.TRANSCENDENT_CRITIQUE] += 0.2
            
        # Buddhist tradition enhances participatory knowing
        if self.state.active_traditions[AxialTradition.BUDDHIST] > 0.5:
            self.state.innovations[AxialInnovation.PARTICIPATORY_KNOWING] += 0.2
            
        # Confucian tradition enhances transformative practices
        if self.state.active_traditions[AxialTradition.CONFUCIAN] > 0.5:
            self.state.innovations[AxialInnovation.TRANSFORMATIVE_PRACTICES] += 0.2
            
        # Second order thinking emerges from combination
        if sum(self.state.active_traditions.values()) > 2.0:
            self.state.innovations[AxialInnovation.SECOND_ORDER_THINKING] += 0.3
            
        # Normalize all innovation values
        for innovation in AxialInnovation:
            self.state.innovations[innovation] = min(
                1.0,
                self.state.innovations[innovation]
            )
            
    def _update_integration(self) -> None:
        """Update integration level based on traditions and innovations"""
        tradition_factor = sum(self.state.active_traditions.values()) / len(AxialTradition)
        innovation_factor = sum(self.state.innovations.values()) / len(AxialInnovation)
        
        self.state.integration_level = min(
            1.0,
            0.4 * tradition_factor + 0.6 * innovation_factor
        )
        
    def _update_transformation(self) -> None:
        """Update transformation depth based on innovations and integration"""
        key_innovations = [
            AxialInnovation.PARTICIPATORY_KNOWING,
            AxialInnovation.TRANSFORMATIVE_PRACTICES,
            AxialInnovation.TRANSCENDENT_CRITIQUE
        ]
        
        innovation_depth = sum(
            self.state.innovations[innovation]
            for innovation in key_innovations
        ) / len(key_innovations)
        
        self.state.transformation_depth = min(
            1.0,
            0.6 * innovation_depth + 0.4 * self.state.integration_level
        )
        
    def evaluate_state(self) -> Dict[str, float]:
        """Evaluate the current state of Axial development"""
        return {
            "tradition_activation": sum(self.state.active_traditions.values()) / len(AxialTradition),
            "innovation_development": sum(self.state.innovations.values()) / len(AxialInnovation),
            "integration_level": self.state.integration_level,
            "transformation_depth": self.state.transformation_depth,
            "overall_development": (
                0.25 * sum(self.state.active_traditions.values()) / len(AxialTradition) +
                0.25 * sum(self.state.innovations.values()) / len(AxialInnovation) +
                0.25 * self.state.integration_level +
                0.25 * self.state.transformation_depth
            )
        } 