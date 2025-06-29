from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from .wisdom_core import WisdomCore
from .rationality_core import RationalityCore
from .cognitive_science import CognitiveCore
from .phenomenology import PhenomenologyCore

class PsychotechnologyType(Enum):
    """Types of psychotechnologies in wisdom ecology"""
    INFERENCE = "inference"  # Logical reasoning
    INSIGHT = "insight"      # Pattern recognition
    INTUITION = "intuition"  # Implicit processing
    INTERNALIZATION = "internalization"  # Perspective taking
    UNDERSTANDING = "understanding"  # Grasping significance
    GNOSIS = "gnosis"       # Transformative knowing
    ASPIRATION = "aspiration"  # Value development

class OptimizationMode(Enum):
    """Modes of optimization in wisdom ecology"""
    HORIZONTAL = "horizontal"  # Cross-domain integration
    VERTICAL = "vertical"      # Depth of processing
    RECURSIVE = "recursive"    # Self-organizing dynamics

@dataclass
class EcologyState:
    """State of the wisdom ecology"""
    active_psychotechnologies: Dict[PsychotechnologyType, float]
    optimization_modes: Dict[OptimizationMode, float]
    dynamic_constraints: List[Dict]
    emergence_patterns: List[Dict]
    integration_level: float

class WisdomEcology:
    """Handles wisdom as an ecology of psychotechnologies.
    
    Implements Vervaeke's model of wisdom as a dynamic system of
    mutually constraining and optimizing cognitive processes.
    """
    
    def __init__(
        self,
        wisdom_core: WisdomCore,
        rationality_core: RationalityCore,
        cognitive_core: CognitiveCore,
        phenomenology_core: PhenomenologyCore
    ):
        self.wisdom_core = wisdom_core
        self.rationality_core = rationality_core
        self.cognitive_core = cognitive_core
        self.phenomenology_core = phenomenology_core
        
        self.ecology_state = EcologyState(
            active_psychotechnologies={tech: 0.0 for tech in PsychotechnologyType},
            optimization_modes={mode: 0.0 for mode in OptimizationMode},
            dynamic_constraints=[],
            emergence_patterns=[],
            integration_level=0.0
        )
        
    def activate_psychotechnology(
        self,
        tech_type: PsychotechnologyType,
        intensity: float = 0.5
    ) -> None:
        """Activate a psychotechnology within the ecology."""
        self.ecology_state.active_psychotechnologies[tech_type] = min(
            1.0,
            self.ecology_state.active_psychotechnologies[tech_type] + intensity
        )
        self._update_constraints()
        self._process_emergence()
        
    def optimize_mode(
        self,
        mode: OptimizationMode,
        intensity: float = 0.5
    ) -> None:
        """Optimize a particular mode within the ecology."""
        self.ecology_state.optimization_modes[mode] = min(
            1.0,
            self.ecology_state.optimization_modes[mode] + intensity
        )
        self._update_integration()
        
    def _update_constraints(self) -> None:
        """Update dynamic constraints between psychotechnologies."""
        new_constraints = []
        
        # Inference constrains insight
        if (self.ecology_state.active_psychotechnologies[PsychotechnologyType.INFERENCE] > 0.7 and
            self.ecology_state.active_psychotechnologies[PsychotechnologyType.INSIGHT] > 0.7):
            new_constraints.append({
                "source": PsychotechnologyType.INFERENCE,
                "target": PsychotechnologyType.INSIGHT,
                "type": "mutual_enhancement",
                "strength": 0.8
            })
            
        # Understanding enhances gnosis
        if (self.ecology_state.active_psychotechnologies[PsychotechnologyType.UNDERSTANDING] > 0.6 and
            self.ecology_state.active_psychotechnologies[PsychotechnologyType.GNOSIS] > 0.4):
            new_constraints.append({
                "source": PsychotechnologyType.UNDERSTANDING,
                "target": PsychotechnologyType.GNOSIS,
                "type": "facilitation",
                "strength": 0.7
            })
            
        self.ecology_state.dynamic_constraints = new_constraints
        
    def _process_emergence(self) -> None:
        """Process emergent patterns in the ecology."""
        new_patterns = []
        
        # Check for sophrosyne emergence
        if self._check_sophrosyne_conditions():
            new_patterns.append({
                "type": "sophrosyne",
                "components": [
                    PsychotechnologyType.INTERNALIZATION,
                    PsychotechnologyType.UNDERSTANDING
                ],
                "strength": 0.8
            })
            
        # Check for transformative understanding
        if self._check_transformation_conditions():
            new_patterns.append({
                "type": "transformative_understanding",
                "components": [
                    PsychotechnologyType.UNDERSTANDING,
                    PsychotechnologyType.GNOSIS
                ],
                "strength": 0.7
            })
            
        self.ecology_state.emergence_patterns = new_patterns
        
    def _update_integration(self) -> None:
        """Update overall integration level of the ecology."""
        # Compute horizontal integration
        horizontal = self.ecology_state.optimization_modes[OptimizationMode.HORIZONTAL]
        
        # Compute vertical integration
        vertical = self.ecology_state.optimization_modes[OptimizationMode.VERTICAL]
        
        # Compute recursive optimization
        recursive = self.ecology_state.optimization_modes[OptimizationMode.RECURSIVE]
        
        # Update overall integration
        self.ecology_state.integration_level = (
            0.4 * horizontal +
            0.4 * vertical +
            0.2 * recursive
        )
        
    def _check_sophrosyne_conditions(self) -> bool:
        """Check conditions for sophrosyne emergence."""
        return (
            self.ecology_state.active_psychotechnologies[PsychotechnologyType.INTERNALIZATION] > 0.7 and
            self.ecology_state.active_psychotechnologies[PsychotechnologyType.UNDERSTANDING] > 0.7 and
            self.ecology_state.optimization_modes[OptimizationMode.VERTICAL] > 0.6
        )
        
    def _check_transformation_conditions(self) -> bool:
        """Check conditions for transformative understanding."""
        return (
            self.ecology_state.active_psychotechnologies[PsychotechnologyType.UNDERSTANDING] > 0.6 and
            self.ecology_state.active_psychotechnologies[PsychotechnologyType.GNOSIS] > 0.6 and
            self.ecology_state.optimization_modes[OptimizationMode.RECURSIVE] > 0.5
        )
        
    def evaluate_ecology(self) -> Dict[str, float]:
        """Evaluate the current state of the wisdom ecology."""
        active_techs = sum(
            1.0 for level in self.ecology_state.active_psychotechnologies.values()
            if level > 0.5
        ) / len(PsychotechnologyType)
        
        constraints = len(self.ecology_state.dynamic_constraints) / 10.0
        emergence = len(self.ecology_state.emergence_patterns) / 5.0
        
        return {
            "active_technologies": active_techs,
            "dynamic_constraints": constraints,
            "emergence_patterns": emergence,
            "integration_level": self.ecology_state.integration_level,
            "overall_ecology": (
                0.3 * active_techs +
                0.2 * constraints +
                0.2 * emergence +
                0.3 * self.ecology_state.integration_level
            )
        }
        
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for optimizing the wisdom ecology."""
        recommendations = []
        
        # Check for weak psychotechnologies
        weak_techs = [
            tech for tech, level in self.ecology_state.active_psychotechnologies.items()
            if level < 0.3
        ]
        if weak_techs:
            recommendations.append(
                f"Develop psychotechnologies: {', '.join(t.value for t in weak_techs)}"
            )
            
        # Check optimization modes
        weak_modes = [
            mode for mode, level in self.ecology_state.optimization_modes.items()
            if level < 0.3
        ]
        if weak_modes:
            recommendations.append(
                f"Enhance optimization modes: {', '.join(m.value for m in weak_modes)}"
            )
            
        # Check integration
        if self.ecology_state.integration_level < 0.5:
            recommendations.append(
                "Focus on integrating active psychotechnologies through recursive optimization"
            )
            
        return recommendations 