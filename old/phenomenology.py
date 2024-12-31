from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class PhenomenologicalMode(Enum):
    """Modes of phenomenological investigation"""
    REFLECTIVE = "reflective"
    EXPERIMENTAL = "experimental"
    EXPLORATORY = "exploratory"
    PROBATIVE = "probative"

class ExistentialDimension(Enum):
    """Dimensions of existential investigation"""
    BEING = "being"
    DASEIN = "dasein"
    INTENTIONALITY = "intentionality"
    WORLD_DISCLOSURE = "world_disclosure"
    PARTICIPATORY = "participatory"

@dataclass
class PhenomenologicalState:
    """State of phenomenological investigation"""
    active_modes: Dict[PhenomenologicalMode, float]
    existential_dimensions: Dict[ExistentialDimension, float]
    intentional_relations: List[Dict]
    world_disclosures: List[Dict]
    participatory_depth: float

class PhenomenologyCore:
    """Handles phenomenological investigation and existential dimensions.
    
    Implements insights from Husserl and Heidegger about phenomenology,
    existential investigation, and participatory knowing.
    """
    
    def __init__(self):
        self.phenomenological_state = PhenomenologicalState(
            active_modes={mode: 0.0 for mode in PhenomenologicalMode},
            existential_dimensions={dim: 0.0 for dim in ExistentialDimension},
            intentional_relations=[],
            world_disclosures=[],
            participatory_depth=0.0
        )
        
    def activate_mode(
        self,
        mode: PhenomenologicalMode,
        intensity: float = 0.5
    ) -> None:
        """Activate a phenomenological mode."""
        self.phenomenological_state.active_modes[mode] = min(
            1.0,
            self.phenomenological_state.active_modes[mode] + intensity
        )
        self._update_relations()
        self._process_disclosures()
        
    def explore_dimension(
        self,
        dimension: ExistentialDimension,
        intensity: float = 0.5
    ) -> None:
        """Explore an existential dimension."""
        self.phenomenological_state.existential_dimensions[dimension] = min(
            1.0,
            self.phenomenological_state.existential_dimensions[dimension] + intensity
        )
        self._update_participatory()
        
    def _update_relations(self) -> None:
        """Update intentional relations in phenomenological investigation."""
        new_relations = []
        
        # Reflective mode enhances intentionality
        if (self.phenomenological_state.active_modes[PhenomenologicalMode.REFLECTIVE] > 0.7 and
            self.phenomenological_state.existential_dimensions[ExistentialDimension.INTENTIONALITY] > 0.6):
            new_relations.append({
                "type": "noetic_enhancement",
                "source": PhenomenologicalMode.REFLECTIVE,
                "target": ExistentialDimension.INTENTIONALITY,
                "strength": 0.8
            })
            
        # Exploratory mode enhances world disclosure
        if (self.phenomenological_state.active_modes[PhenomenologicalMode.EXPLORATORY] > 0.6 and
            self.phenomenological_state.existential_dimensions[ExistentialDimension.WORLD_DISCLOSURE] > 0.5):
            new_relations.append({
                "type": "world_opening",
                "source": PhenomenologicalMode.EXPLORATORY,
                "target": ExistentialDimension.WORLD_DISCLOSURE,
                "strength": 0.7
            })
            
        self.phenomenological_state.intentional_relations = new_relations
        
    def _process_disclosures(self) -> None:
        """Process world disclosures in phenomenological investigation."""
        new_disclosures = []
        
        # Check for dasein disclosure
        if self._check_dasein_conditions():
            new_disclosures.append({
                "type": "being_disclosure",
                "components": [
                    ExistentialDimension.DASEIN,
                    ExistentialDimension.BEING
                ],
                "strength": 0.8
            })
            
        # Check for participatory disclosure
        if self._check_participatory_conditions():
            new_disclosures.append({
                "type": "participatory_disclosure",
                "components": [
                    ExistentialDimension.PARTICIPATORY,
                    ExistentialDimension.WORLD_DISCLOSURE
                ],
                "strength": 0.7
            })
            
        self.phenomenological_state.world_disclosures = new_disclosures
        
    def _update_participatory(self) -> None:
        """Update participatory depth in phenomenological investigation."""
        being = self.phenomenological_state.existential_dimensions[ExistentialDimension.BEING]
        dasein = self.phenomenological_state.existential_dimensions[ExistentialDimension.DASEIN]
        participatory = self.phenomenological_state.existential_dimensions[ExistentialDimension.PARTICIPATORY]
        
        self.phenomenological_state.participatory_depth = (
            0.4 * being +
            0.3 * dasein +
            0.3 * participatory
        )
        
    def _check_dasein_conditions(self) -> bool:
        """Check conditions for dasein disclosure."""
        return (
            self.phenomenological_state.existential_dimensions[ExistentialDimension.DASEIN] > 0.7 and
            self.phenomenological_state.existential_dimensions[ExistentialDimension.BEING] > 0.6 and
            self.phenomenological_state.active_modes[PhenomenologicalMode.REFLECTIVE] > 0.6
        )
        
    def _check_participatory_conditions(self) -> bool:
        """Check conditions for participatory disclosure."""
        return (
            self.phenomenological_state.existential_dimensions[ExistentialDimension.PARTICIPATORY] > 0.6 and
            self.phenomenological_state.existential_dimensions[ExistentialDimension.WORLD_DISCLOSURE] > 0.6 and
            self.phenomenological_state.active_modes[PhenomenologicalMode.EXPLORATORY] > 0.5
        )
        
    def evaluate_investigation(self) -> Dict[str, float]:
        """Evaluate the current state of phenomenological investigation."""
        active_modes = sum(
            1.0 for level in self.phenomenological_state.active_modes.values()
            if level > 0.5
        ) / len(PhenomenologicalMode)
        
        dimensions = sum(
            1.0 for level in self.phenomenological_state.existential_dimensions.values()
            if level > 0.5
        ) / len(ExistentialDimension)
        
        relations = len(self.phenomenological_state.intentional_relations) / 10.0
        disclosures = len(self.phenomenological_state.world_disclosures) / 5.0
        
        return {
            "active_modes": active_modes,
            "existential_dimensions": dimensions,
            "intentional_relations": relations,
            "world_disclosures": disclosures,
            "participatory_depth": self.phenomenological_state.participatory_depth,
            "overall_investigation": (
                0.2 * active_modes +
                0.2 * dimensions +
                0.2 * relations +
                0.2 * disclosures +
                0.2 * self.phenomenological_state.participatory_depth
            )
        }
        
    def get_investigation_recommendations(self) -> List[str]:
        """Get recommendations for deepening phenomenological investigation."""
        recommendations = []
        
        weak_modes = [
            mode for mode, level in self.phenomenological_state.active_modes.items()
            if level < 0.3
        ]
        if weak_modes:
            recommendations.append(
                f"Develop phenomenological modes: {', '.join(m.value for m in weak_modes)}"
            )
            
        weak_dims = [
            dim for dim, level in self.phenomenological_state.existential_dimensions.items()
            if level < 0.3
        ]
        if weak_dims:
            recommendations.append(
                f"Explore existential dimensions: {', '.join(d.value for d in weak_dims)}"
            )
            
        if self.phenomenological_state.participatory_depth < 0.5:
            recommendations.append(
                "Deepen participatory knowing through existential investigation"
            )
            
        return recommendations 