from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from ..integration.imaginal import ImaginalCore

class ArchetypeType(Enum):
    """Types of archetypes in Jungian psychology"""
    HERO = "hero"
    SHADOW = "shadow"
    ANIMA = "anima"
    ANIMUS = "animus"
    SELF = "self"
    EGO = "ego"
    WISE_OLD_MAN = "wise_old_man"
    GREAT_MOTHER = "great_mother"

class PsychicDimension(Enum):
    """Dimensions of psychic development"""
    CONSCIOUS = "conscious"
    PERSONAL_UNCONSCIOUS = "personal_unconscious"
    COLLECTIVE_UNCONSCIOUS = "collective_unconscious"
    TRANSCENDENT = "transcendent"

@dataclass
class ArchetypeState:
    """State of archetypal engagement"""
    archetype_type: ArchetypeType
    activation_level: float
    integration_level: float
    constellation_patterns: List[Dict]
    developmental_stage: int

@dataclass
class PsychicState:
    """State of psychic development"""
    active_archetypes: Dict[ArchetypeType, ArchetypeState]
    dimension_activation: Dict[PsychicDimension, float]
    ego_self_axis: float
    individuation_level: float
    integration_patterns: List[Dict]

class PsycheCore:
    """Handles psychic development and archetypal engagement.
    
    Implements Jung's understanding of the psyche
    and its role in individuation.
    """
    
    def __init__(
        self,
        imaginal_core: ImaginalCore
    ):
        self.imaginal_core = imaginal_core
        
        self.psychic_state = PsychicState(
            active_archetypes={
                archetype: ArchetypeState(
                    archetype_type=archetype,
                    activation_level=0.0,
                    integration_level=0.0,
                    constellation_patterns=[],
                    developmental_stage=0
                ) for archetype in ArchetypeType
            },
            dimension_activation={dim: 0.0 for dim in PsychicDimension},
            ego_self_axis=0.0,
            individuation_level=0.0,
            integration_patterns=[]
        )
        
    def activate_archetype(
        self,
        archetype: ArchetypeType,
        intensity: float = 0.5
    ) -> None:
        """Activate an archetypal pattern."""
        state = self.psychic_state.active_archetypes[archetype]
        state.activation_level = min(
            1.0,
            state.activation_level + intensity
        )
        self._update_constellation(archetype)
        self._process_integration()
        
    def engage_dimension(
        self,
        dimension: PsychicDimension,
        intensity: float = 0.5
    ) -> None:
        """Engage a psychic dimension."""
        self.psychic_state.dimension_activation[dimension] = min(
            1.0,
            self.psychic_state.dimension_activation[dimension] + intensity
        )
        self._update_ego_self_axis()
        
    def _update_constellation(
        self,
        archetype: ArchetypeType
    ) -> None:
        """Update constellation patterns for an archetype."""
        state = self.psychic_state.active_archetypes[archetype]
        new_patterns = []
        
        # Hero-Shadow constellation
        if (archetype == ArchetypeType.HERO and
            self.psychic_state.active_archetypes[ArchetypeType.SHADOW].activation_level > 0.6):
            new_patterns.append({
                "type": "hero_shadow_integration",
                "components": [ArchetypeType.HERO, ArchetypeType.SHADOW],
                "strength": 0.8
            })
            
        # Anima-Animus constellation
        if (archetype in [ArchetypeType.ANIMA, ArchetypeType.ANIMUS] and
            self.psychic_state.active_archetypes[ArchetypeType.SELF].activation_level > 0.7):
            new_patterns.append({
                "type": "contrasexual_integration",
                "components": [ArchetypeType.ANIMA, ArchetypeType.ANIMUS],
                "strength": 0.7
            })
            
        state.constellation_patterns = new_patterns
        
    def _process_integration(self) -> None:
        """Process integration patterns in psychic development."""
        new_patterns = []
        
        # Check for individuation progress
        if self._check_individuation_conditions():
            new_patterns.append({
                "type": "individuation_progress",
                "components": [
                    ArchetypeType.EGO,
                    ArchetypeType.SELF
                ],
                "strength": 0.8
            })
            
        # Check for transcendent function
        if self._check_transcendent_conditions():
            new_patterns.append({
                "type": "transcendent_function",
                "components": [
                    PsychicDimension.CONSCIOUS,
                    PsychicDimension.COLLECTIVE_UNCONSCIOUS
                ],
                "strength": 0.7
            })
            
        self.psychic_state.integration_patterns = new_patterns
        
    def _update_ego_self_axis(self) -> None:
        """Update the ego-Self axis strength."""
        conscious = self.psychic_state.dimension_activation[PsychicDimension.CONSCIOUS]
        collective = self.psychic_state.dimension_activation[PsychicDimension.COLLECTIVE_UNCONSCIOUS]
        transcendent = self.psychic_state.dimension_activation[PsychicDimension.TRANSCENDENT]
        
        self.psychic_state.ego_self_axis = (
            0.3 * conscious +
            0.3 * collective +
            0.4 * transcendent
        )
        
    def _check_individuation_conditions(self) -> bool:
        """Check conditions for individuation progress."""
        return (
            self.psychic_state.active_archetypes[ArchetypeType.EGO].activation_level > 0.7 and
            self.psychic_state.active_archetypes[ArchetypeType.SELF].activation_level > 0.6 and
            self.psychic_state.ego_self_axis > 0.6
        )
        
    def _check_transcendent_conditions(self) -> bool:
        """Check conditions for transcendent function."""
        return (
            self.psychic_state.dimension_activation[PsychicDimension.CONSCIOUS] > 0.6 and
            self.psychic_state.dimension_activation[PsychicDimension.COLLECTIVE_UNCONSCIOUS] > 0.6 and
            self.psychic_state.ego_self_axis > 0.5
        )
        
    def evaluate_development(self) -> Dict[str, float]:
        """Evaluate the current state of psychic development."""
        active_archetypes = sum(
            1.0 for state in self.psychic_state.active_archetypes.values()
            if state.activation_level > 0.5
        ) / len(ArchetypeType)
        
        active_dimensions = sum(
            1.0 for level in self.psychic_state.dimension_activation.values()
            if level > 0.5
        ) / len(PsychicDimension)
        
        constellation = len(sum(
            [state.constellation_patterns for state in self.psychic_state.active_archetypes.values()],
            []
        )) / 10.0
        
        integration = len(self.psychic_state.integration_patterns) / 5.0
        
        return {
            "active_archetypes": active_archetypes,
            "active_dimensions": active_dimensions,
            "constellation_patterns": constellation,
            "integration_patterns": integration,
            "ego_self_axis": self.psychic_state.ego_self_axis,
            "overall_development": (
                0.2 * active_archetypes +
                0.2 * active_dimensions +
                0.2 * constellation +
                0.2 * integration +
                0.2 * self.psychic_state.ego_self_axis
            )
        }
        
    def get_development_recommendations(self) -> List[str]:
        """Get recommendations for deepening psychic development."""
        recommendations = []
        
        # Check for weak archetypes
        weak_archetypes = [
            archetype for archetype, state in self.psychic_state.active_archetypes.items()
            if state.activation_level < 0.3
        ]
        if weak_archetypes:
            recommendations.append(
                f"Engage archetypes: {', '.join(a.value for a in weak_archetypes)}"
            )
            
        # Check dimensions
        weak_dims = [
            dim for dim, level in self.psychic_state.dimension_activation.items()
            if level < 0.3
        ]
        if weak_dims:
            recommendations.append(
                f"Develop psychic dimensions: {', '.join(d.value for d in weak_dims)}"
            )
            
        # Check ego-Self axis
        if self.psychic_state.ego_self_axis < 0.5:
            recommendations.append(
                "Strengthen ego-Self axis through balanced archetypal engagement"
            )
            
        return recommendations 