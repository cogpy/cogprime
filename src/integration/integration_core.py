from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from ..integration.aletheia import AletheiaCore
from ..integration.imaginal import ImaginalCore
from ..integration.psyche import PsycheCore
from ..cognitive_science.phenomenology_core import PhenomenologyCore
from .wisdom_ecology import WisdomEcology

class IntegrationMode(Enum):
    """Modes of integration in final participation"""
    POESIS = "poesis"  # Creative participation
    GELASSENHEIT = "gelassenheit"  # Letting be
    DURCHBROCK = "durchbrock"  # Breakthrough
    EPEKTASIS = "epektasis"  # Infinite striving
    SYNERGISTIC = "synergistic"  # Co-creative engagement
    IMAGINAL = "imaginal"  # Bridging abstract and concrete
    ASPIRATIONAL = "aspirational"  # Moving toward divine double

class NonTheisticDimension(Enum):
    """Dimensions of non-theistic engagement"""
    GROUND_OF_BEING = "ground_of_being"  # Ultimate reality
    INEXHAUSTIBLE = "inexhaustible"  # Infinite potential
    PARTICIPATORY = "participatory"  # Co-creative engagement
    TRANSJECTIVE = "transjective"  # Beyond subject-object
    TRANSFORMATIVE = "transformative"  # Self-transcending
    IMAGINAL = "imaginal"  # Bridging worlds
    DIVINE_DOUBLE = "divine_double"  # Aspirational self

@dataclass
class ImaginalBridge:
    """Bridge between abstract and concrete worlds"""
    abstract_pole: str
    concrete_pole: str
    mediating_symbol: str
    bridge_strength: float
    transformative_potential: float

@dataclass
class DivineDouble:
    """Representation of aspirational self"""
    current_self: Dict[str, float]  # Current state
    aspirational_self: Dict[str, float]  # Target state
    bridge_symbols: List[str]  # Mediating symbols
    non_logical_identities: List[str]  # Identity patterns
    transformation_trajectory: float  # Progress

@dataclass
class IntegrationState:
    """State of final integration"""
    active_modes: Dict[IntegrationMode, float]
    nontheistic_dimensions: Dict[NonTheisticDimension, float]
    creative_patterns: List[Dict]
    breakthrough_patterns: List[Dict]
    synergistic_depth: float
    imaginal_bridges: List[ImaginalBridge]
    divine_doubles: List[DivineDouble]

class IntegrationCore:
    """Handles final integration of all frameworks.
    
    Implements Vervaeke's synthesis of frameworks for addressing
    the meaning crisis through non-theistic participation.
    """
    
    def __init__(
        self,
        aletheia_core: AletheiaCore,
        imaginal_core: ImaginalCore,
        psyche_core: PsycheCore,
        phenomenology_core: PhenomenologyCore,
        wisdom_ecology: WisdomEcology
    ):
        self.aletheia_core = aletheia_core
        self.imaginal_core = imaginal_core
        self.psyche_core = psyche_core
        self.phenomenology_core = phenomenology_core
        self.wisdom_ecology = wisdom_ecology
        
        self.integration_state = IntegrationState(
            active_modes={mode: 0.0 for mode in IntegrationMode},
            nontheistic_dimensions={dim: 0.0 for dim in NonTheisticDimension},
            creative_patterns=[],
            breakthrough_patterns=[],
            synergistic_depth=0.0,
            imaginal_bridges=[],
            divine_doubles=[]
        )
        
    def create_imaginal_bridge(
        self,
        abstract_pole: str,
        concrete_pole: str,
        mediating_symbol: str,
        initial_strength: float = 0.5
    ) -> None:
        """Create a bridge between abstract and concrete worlds."""
        bridge = ImaginalBridge(
            abstract_pole=abstract_pole,
            concrete_pole=concrete_pole,
            mediating_symbol=mediating_symbol,
            bridge_strength=initial_strength,
            transformative_potential=initial_strength * 0.8
        )
        self.integration_state.imaginal_bridges.append(bridge)
        self._update_imaginal_resonance()

    def initiate_divine_double(
        self,
        current_state: Dict[str, float],
        aspirational_state: Dict[str, float],
        bridge_symbols: List[str]
    ) -> None:
        """Initialize a divine double aspiration."""
        double = DivineDouble(
            current_self=current_state,
            aspirational_self=aspirational_state,
            bridge_symbols=bridge_symbols,
            non_logical_identities=[],
            transformation_trajectory=0.0
        )
        self.integration_state.divine_doubles.append(double)
        self._update_aspirational_dynamics()

    def _update_imaginal_resonance(self) -> None:
        """Update resonance patterns in imaginal bridges."""
        for bridge in self.integration_state.imaginal_bridges:
            # Enhance bridge strength through symbolic resonance
            if self.integration_state.active_modes[IntegrationMode.IMAGINAL] > 0.6:
                bridge.bridge_strength = min(
                    1.0,
                    bridge.bridge_strength + 0.1
                )
                bridge.transformative_potential = min(
                    1.0,
                    bridge.transformative_potential + 0.15
                )

    def _update_aspirational_dynamics(self) -> None:
        """Update transformation trajectories in divine doubles."""
        for double in self.integration_state.divine_doubles:
            # Calculate trajectory based on bridge symbols and non-logical identities
            symbol_factor = len(double.bridge_symbols) / 10.0
            identity_factor = len(double.non_logical_identities) / 5.0
            
            double.transformation_trajectory = min(
                1.0,
                0.3 * symbol_factor +
                0.3 * identity_factor +
                0.4 * self.integration_state.active_modes[IntegrationMode.ASPIRATIONAL]
            )

    def activate_mode(
        self,
        mode: IntegrationMode,
        intensity: float = 0.5
    ) -> None:
        """Activate an integration mode with cross-framework resonance."""
        self.integration_state.active_modes[mode] = min(
            1.0,
            self.integration_state.active_modes[mode] + intensity
        )
        self._update_creativity()
        self._process_breakthrough()
        
        # Integrate across frameworks
        if mode == IntegrationMode.POESIS:
            self.imaginal_core.activate_mode("transformative", intensity * 0.8)
            self.psyche_core.activate_archetype("self", intensity * 0.7)
        elif mode == IntegrationMode.EPEKTASIS:
            self.aletheia_core.activate_truth_mode("aletheia", intensity * 0.8)
            self.phenomenology_core.explore_dimension("being", intensity * 0.7)
        elif mode == IntegrationMode.IMAGINAL:
            self._update_imaginal_resonance()
        elif mode == IntegrationMode.ASPIRATIONAL:
            self._update_aspirational_dynamics()
        
    def engage_dimension(
        self,
        dimension: NonTheisticDimension,
        intensity: float = 0.5
    ) -> None:
        """Engage a non-theistic dimension with cross-framework resonance."""
        self.integration_state.nontheistic_dimensions[dimension] = min(
            1.0,
            self.integration_state.nontheistic_dimensions[dimension] + intensity
        )
        self._update_synergy()
        
        # Integrate across frameworks
        if dimension == NonTheisticDimension.GROUND_OF_BEING:
            self.aletheia_core.engage_being_mode("being", intensity * 0.8)
            self.wisdom_ecology.activate_psychotechnology("gnosis", intensity * 0.7)
        elif dimension == NonTheisticDimension.PARTICIPATORY:
            self.phenomenology_core.explore_dimension("participatory", intensity * 0.8)
            self.imaginal_core.engage_dimension("bridging", intensity * 0.7)
        elif dimension == NonTheisticDimension.IMAGINAL:
            self._update_imaginal_resonance()
        elif dimension == NonTheisticDimension.DIVINE_DOUBLE:
            self._update_aspirational_dynamics()
        
    def _update_creativity(self) -> None:
        """Update creative patterns in integration."""
        new_patterns = []
        
        # Poesis enhances transformation
        if (self.integration_state.active_modes[IntegrationMode.POESIS] > 0.7 and
            self.integration_state.nontheistic_dimensions[NonTheisticDimension.TRANSFORMATIVE] > 0.6):
            new_patterns.append({
                "type": "creative_transformation",
                "source": IntegrationMode.POESIS,
                "target": NonTheisticDimension.TRANSFORMATIVE,
                "strength": 0.8,
                "cross_resonance": True
            })
            
        # Synergistic enhances participation
        if (self.integration_state.active_modes[IntegrationMode.SYNERGISTIC] > 0.6 and
            self.integration_state.nontheistic_dimensions[NonTheisticDimension.PARTICIPATORY] > 0.5):
            new_patterns.append({
                "type": "participatory_creation",
                "source": IntegrationMode.SYNERGISTIC,
                "target": NonTheisticDimension.PARTICIPATORY,
                "strength": 0.7,
                "cross_resonance": True
            })
            
        self.integration_state.creative_patterns = new_patterns
        
    def _process_breakthrough(self) -> None:
        """Process breakthrough patterns in integration."""
        new_patterns = []
        
        # Check for epektasis breakthrough
        if self._check_epektasis_conditions():
            new_patterns.append({
                "type": "infinite_striving",
                "components": [
                    IntegrationMode.EPEKTASIS,
                    NonTheisticDimension.INEXHAUSTIBLE
                ],
                "strength": 0.8,
                "cross_resonance": True
            })
            
        # Check for durchbrock breakthrough
        if self._check_durchbrock_conditions():
            new_patterns.append({
                "type": "transformative_breakthrough",
                "components": [
                    IntegrationMode.DURCHBROCK,
                    NonTheisticDimension.TRANSFORMATIVE
                ],
                "strength": 0.7,
                "cross_resonance": True
            })
            
        self.integration_state.breakthrough_patterns = new_patterns
        
    def _update_synergy(self) -> None:
        """Update synergistic depth in integration."""
        ground = self.integration_state.nontheistic_dimensions[NonTheisticDimension.GROUND_OF_BEING]
        participatory = self.integration_state.nontheistic_dimensions[NonTheisticDimension.PARTICIPATORY]
        transformative = self.integration_state.nontheistic_dimensions[NonTheisticDimension.TRANSFORMATIVE]
        
        # Include cross-framework resonance
        framework_factor = (
            self.aletheia_core.aletheia_state.truth_modes["aletheia"] +
            self.imaginal_core.imaginal_state.active_modes["transformative"] +
            self.psyche_core.psychic_state.ego_self_axis +
            self.phenomenology_core.phenomenological_state.participatory_depth +
            self.wisdom_ecology.ecology_state.integration_level
        ) / 5.0
        
        self.integration_state.synergistic_depth = (
            0.3 * ground +
            0.3 * participatory +
            0.3 * transformative +
            0.1 * framework_factor
        )
        
    def _check_epektasis_conditions(self) -> bool:
        """Check conditions for epektasis breakthrough."""
        base_conditions = (
            self.integration_state.active_modes[IntegrationMode.EPEKTASIS] > 0.7 and
            self.integration_state.nontheistic_dimensions[NonTheisticDimension.INEXHAUSTIBLE] > 0.6 and
            self.integration_state.active_modes[IntegrationMode.POESIS] > 0.6
        )
        
        framework_conditions = (
            self.aletheia_core.aletheia_state.truth_modes["aletheia"] > 0.5 and
            self.imaginal_core.imaginal_state.active_modes["transformative"] > 0.5 and
            self.psyche_core.psychic_state.ego_self_axis > 0.5
        )
        
        return base_conditions and framework_conditions
        
    def _check_durchbrock_conditions(self) -> bool:
        """Check conditions for durchbrock breakthrough."""
        base_conditions = (
            self.integration_state.active_modes[IntegrationMode.DURCHBROCK] > 0.6 and
            self.integration_state.nontheistic_dimensions[NonTheisticDimension.TRANSFORMATIVE] > 0.6 and
            self.integration_state.active_modes[IntegrationMode.GELASSENHEIT] > 0.5
        )
        
        framework_conditions = (
            self.phenomenology_core.phenomenological_state.participatory_depth > 0.5 and
            self.wisdom_ecology.ecology_state.integration_level > 0.5
        )
        
        return base_conditions and framework_conditions
        
    def evaluate_integration(self) -> Dict[str, float]:
        """Evaluate the current state of integration."""
        active_modes = sum(
            1.0 for level in self.integration_state.active_modes.values()
            if level > 0.5
        ) / len(IntegrationMode)
        
        dimensions = sum(
            1.0 for level in self.integration_state.nontheistic_dimensions.values()
            if level > 0.5
        ) / len(NonTheisticDimension)
        
        creativity = len(self.integration_state.creative_patterns) / 10.0
        breakthrough = len(self.integration_state.breakthrough_patterns) / 5.0
        
        imaginal_factor = sum(
            bridge.bridge_strength for bridge in self.integration_state.imaginal_bridges
        ) / max(len(self.integration_state.imaginal_bridges), 1)
        
        aspirational_factor = sum(
            double.transformation_trajectory 
            for double in self.integration_state.divine_doubles
        ) / max(len(self.integration_state.divine_doubles), 1)
        
        # Include cross-framework resonance
        framework_factor = (
            self.aletheia_core.evaluate_disclosure()["overall_disclosure"] +
            self.imaginal_core.evaluate_engagement()["overall_engagement"] +
            self.psyche_core.evaluate_development()["overall_development"] +
            self.phenomenology_core.evaluate_investigation()["overall_investigation"] +
            self.wisdom_ecology.evaluate_ecology()["overall_ecology"]
        ) / 5.0
        
        return {
            "active_modes": active_modes,
            "nontheistic_dimensions": dimensions,
            "creative_patterns": creativity,
            "breakthrough_patterns": breakthrough,
            "synergistic_depth": self.integration_state.synergistic_depth,
            "imaginal_bridging": imaginal_factor,
            "aspirational_trajectory": aspirational_factor,
            "framework_resonance": framework_factor,
            "overall_integration": (
                0.15 * active_modes +
                0.15 * dimensions +
                0.15 * creativity +
                0.15 * breakthrough +
                0.1 * self.integration_state.synergistic_depth +
                0.1 * imaginal_factor +
                0.1 * aspirational_factor +
                0.1 * framework_factor
            )
        }
        
    def get_integration_recommendations(self) -> List[str]:
        """Get recommendations for deepening integration."""
        recommendations = []
        
        # Check for weak modes
        weak_modes = [
            mode for mode, level in self.integration_state.active_modes.items()
            if level < 0.3
        ]
        if weak_modes:
            recommendations.append(
                f"Develop integration modes: {', '.join(m.value for m in weak_modes)}"
            )
            
        # Check dimensions
        weak_dims = [
            dim for dim, level in self.integration_state.nontheistic_dimensions.items()
            if level < 0.3
        ]
        if weak_dims:
            recommendations.append(
                f"Engage non-theistic dimensions: {', '.join(d.value for d in weak_dims)}"
            )
            
        # Check synergistic depth
        if self.integration_state.synergistic_depth < 0.5:
            recommendations.append(
                "Deepen synergistic engagement through balanced participation"
            )
            
        # Check imaginal bridges
        if not self.integration_state.imaginal_bridges:
            recommendations.append(
                "Create imaginal bridges between abstract and concrete worlds"
            )
        elif any(bridge.bridge_strength < 0.4 for bridge in self.integration_state.imaginal_bridges):
            recommendations.append(
                "Strengthen existing imaginal bridges through symbolic resonance"
            )
            
        # Check divine doubles
        if not self.integration_state.divine_doubles:
            recommendations.append(
                "Initiate divine double aspirations for transformative development"
            )
        elif any(double.transformation_trajectory < 0.4 for double in self.integration_state.divine_doubles):
            recommendations.append(
                "Enhance aspirational trajectories through symbolic mediation"
            )
            
        # Check framework resonance
        framework_factor = (
            self.aletheia_core.evaluate_disclosure()["overall_disclosure"] +
            self.imaginal_core.evaluate_engagement()["overall_engagement"] +
            self.psyche_core.evaluate_development()["overall_development"] +
            self.phenomenology_core.evaluate_investigation()["overall_investigation"] +
            self.wisdom_ecology.evaluate_ecology()["overall_ecology"]
        ) / 5.0
        
        if framework_factor < 0.5:
            recommendations.append(
                "Enhance cross-framework resonance through integrated practice"
            )
            
        return recommendations 