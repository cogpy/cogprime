from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from .aletheia import AletheiaCore

class ImaginalMode(Enum):
    """Modes of imaginal engagement"""
    ABSTRACT = "abstract"  # Abstract intelligible world
    CONCRETE = "concrete"  # Concrete sensible world
    SUBJECTIVE = "subjective"  # Personal experience
    OBJECTIVE = "objective"  # External reality
    TRANSFORMATIVE = "transformative"  # Dynamic transframing

class AspirationalDimension(Enum):
    """Dimensions of aspirational development"""
    CURRENT_SELF = "current_self"  # Present state
    DIVINE_DOUBLE = "divine_double"  # Aspirational state
    BRIDGING = "bridging"  # Process of transformation
    INTEGRATION = "integration"  # Unity of selves
    TRANSCENDENCE = "transcendence"  # Beyond current frame

@dataclass
class ImaginalState:
    """State of imaginal engagement"""
    active_modes: Dict[ImaginalMode, float]
    aspirational_dimensions: Dict[AspirationalDimension, float]
    bridging_patterns: List[Dict]
    transformation_patterns: List[Dict]
    integration_depth: float

class ImaginalCore:
    """Handles imaginal engagement and aspirational development.
    
    Implements Corbin's understanding of the imaginal realm
    and its role in self-transcendence and gnosis.
    """
    
    def __init__(
        self,
        aletheia_core: AletheiaCore
    ):
        self.aletheia_core = aletheia_core
        
        self.imaginal_state = ImaginalState(
            active_modes={mode: 0.0 for mode in ImaginalMode},
            aspirational_dimensions={dim: 0.0 for dim in AspirationalDimension},
            bridging_patterns=[],
            transformation_patterns=[],
            integration_depth=0.0
        )
        
    def activate_mode(
        self,
        mode: ImaginalMode,
        intensity: float = 0.5
    ) -> None:
        """Activate an imaginal mode."""
        self.imaginal_state.active_modes[mode] = min(
            1.0,
            self.imaginal_state.active_modes[mode] + intensity
        )
        self._update_bridging()
        self._process_transformation()
        
    def engage_dimension(
        self,
        dimension: AspirationalDimension,
        intensity: float = 0.5
    ) -> None:
        """Engage an aspirational dimension."""
        self.imaginal_state.aspirational_dimensions[dimension] = min(
            1.0,
            self.imaginal_state.aspirational_dimensions[dimension] + intensity
        )
        self._update_integration()
        
    def _update_bridging(self) -> None:
        """Update bridging patterns in imaginal engagement."""
        new_patterns = []
        
        # Abstract-Concrete bridging
        if (self.imaginal_state.active_modes[ImaginalMode.ABSTRACT] > 0.7 and
            self.imaginal_state.active_modes[ImaginalMode.CONCRETE] > 0.6):
            new_patterns.append({
                "type": "world_bridging",
                "source": ImaginalMode.ABSTRACT,
                "target": ImaginalMode.CONCRETE,
                "strength": 0.8
            })
            
        # Subjective-Objective bridging
        if (self.imaginal_state.active_modes[ImaginalMode.SUBJECTIVE] > 0.6 and
            self.imaginal_state.active_modes[ImaginalMode.OBJECTIVE] > 0.5):
            new_patterns.append({
                "type": "reality_bridging",
                "source": ImaginalMode.SUBJECTIVE,
                "target": ImaginalMode.OBJECTIVE,
                "strength": 0.7
            })
            
        self.imaginal_state.bridging_patterns = new_patterns
        
    def _process_transformation(self) -> None:
        """Process transformation patterns in imaginal engagement."""
        new_patterns = []
        
        # Check for self-transcendence
        if self._check_transcendence_conditions():
            new_patterns.append({
                "type": "self_transcendence",
                "components": [
                    AspirationalDimension.CURRENT_SELF,
                    AspirationalDimension.DIVINE_DOUBLE
                ],
                "strength": 0.8
            })
            
        # Check for integration
        if self._check_integration_conditions():
            new_patterns.append({
                "type": "self_integration",
                "components": [
                    AspirationalDimension.BRIDGING,
                    AspirationalDimension.INTEGRATION
                ],
                "strength": 0.7
            })
            
        self.imaginal_state.transformation_patterns = new_patterns
        
    def _update_integration(self) -> None:
        """Update integration depth in imaginal engagement."""
        current = self.imaginal_state.aspirational_dimensions[AspirationalDimension.CURRENT_SELF]
        divine = self.imaginal_state.aspirational_dimensions[AspirationalDimension.DIVINE_DOUBLE]
        bridging = self.imaginal_state.aspirational_dimensions[AspirationalDimension.BRIDGING]
        
        self.imaginal_state.integration_depth = (
            0.3 * current +
            0.3 * divine +
            0.4 * bridging
        )
        
    def _check_transcendence_conditions(self) -> bool:
        """Check conditions for self-transcendence."""
        return (
            self.imaginal_state.aspirational_dimensions[AspirationalDimension.CURRENT_SELF] > 0.7 and
            self.imaginal_state.aspirational_dimensions[AspirationalDimension.DIVINE_DOUBLE] > 0.6 and
            self.imaginal_state.active_modes[ImaginalMode.TRANSFORMATIVE] > 0.6
        )
        
    def _check_integration_conditions(self) -> bool:
        """Check conditions for self-integration."""
        return (
            self.imaginal_state.aspirational_dimensions[AspirationalDimension.BRIDGING] > 0.6 and
            self.imaginal_state.aspirational_dimensions[AspirationalDimension.INTEGRATION] > 0.6 and
            self.imaginal_state.active_modes[ImaginalMode.TRANSFORMATIVE] > 0.5
        )
        
    def evaluate_engagement(self) -> Dict[str, float]:
        """Evaluate the current state of imaginal engagement."""
        active_modes = sum(
            1.0 for level in self.imaginal_state.active_modes.values()
            if level > 0.5
        ) / len(ImaginalMode)
        
        active_dimensions = sum(
            1.0 for level in self.imaginal_state.aspirational_dimensions.values()
            if level > 0.5
        ) / len(AspirationalDimension)
        
        bridging = len(self.imaginal_state.bridging_patterns) / 10.0
        transformation = len(self.imaginal_state.transformation_patterns) / 5.0
        
        return {
            "active_modes": active_modes,
            "active_dimensions": active_dimensions,
            "bridging_patterns": bridging,
            "transformation_patterns": transformation,
            "integration_depth": self.imaginal_state.integration_depth,
            "overall_engagement": (
                0.2 * active_modes +
                0.2 * active_dimensions +
                0.2 * bridging +
                0.2 * transformation +
                0.2 * self.imaginal_state.integration_depth
            )
        }
        
    def get_engagement_recommendations(self) -> List[str]:
        """Get recommendations for deepening imaginal engagement."""
        recommendations = []
        
        # Check for weak modes
        weak_modes = [
            mode for mode, level in self.imaginal_state.active_modes.items()
            if level < 0.3
        ]
        if weak_modes:
            recommendations.append(
                f"Develop imaginal modes: {', '.join(m.value for m in weak_modes)}"
            )
            
        # Check dimensions
        weak_dims = [
            dim for dim, level in self.imaginal_state.aspirational_dimensions.items()
            if level < 0.3
        ]
        if weak_dims:
            recommendations.append(
                f"Engage aspirational dimensions: {', '.join(d.value for d in weak_dims)}"
            )
            
        # Check integration depth
        if self.imaginal_state.integration_depth < 0.5:
            recommendations.append(
                "Deepen integration through balanced bridging and transformation"
            )
            
        return recommendations 