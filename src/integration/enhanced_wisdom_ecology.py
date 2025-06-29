from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from ..cognitive_science.wisdom_core import WisdomCore
from ..cognitive_science.rationality_core import RationalityCore
from ..cognitive_science.cognitive_science import CognitiveCore
from ..cognitive_science.phenomenology import PhenomenologyCore

class PsychotechnologyType(Enum):
    """Types of psychotechnologies in wisdom ecology"""
    INFERENCE = "inference"  # Logical reasoning
    INSIGHT = "insight"      # Pattern recognition
    INTUITION = "intuition"  # Implicit processing
    INTERNALIZATION = "internalization"  # Perspective taking
    UNDERSTANDING = "understanding"  # Grasping significance
    GNOSIS = "gnosis"       # Transformative knowing
    ASPIRATION = "aspiration"  # Value development
    COURAGE = "courage"     # Facing uncertainty
    PRESENCE = "presence"   # Being authentically
    BECOMING = "becoming"   # Dynamic transformation

class OptimizationMode(Enum):
    """Modes of optimization in wisdom ecology"""
    HORIZONTAL = "horizontal"  # Cross-domain integration
    VERTICAL = "vertical"      # Depth of processing
    RECURSIVE = "recursive"    # Self-organizing dynamics
    DIALOGICAL = "dialogical"  # Conversational emergence
    PARTICIPATORY = "participatory"  # Active engagement
    TRANSFORMATIVE = "transformative"  # Deep change

@dataclass
class MeaningStructure:
    """Structure of meaning in wisdom ecology"""
    purpose: float  # Sense of purpose
    coherence: float  # Narrative coherence
    significance: float  # Personal significance
    connectedness: float  # Relational bonds
    transcendence: float  # Self-transcendence
    authenticity: float  # Being true to self

@dataclass
class TransformationPattern:
    """Pattern of transformation in wisdom ecology"""
    source_state: Dict[str, float]  # Initial conditions
    target_state: Dict[str, float]  # Aspired conditions
    bridge_practices: List[PsychotechnologyType]  # Bridging practices
    emergence_factors: List[str]  # Emergent qualities
    integration_level: float  # Pattern coherence

@dataclass
class EcologyState:
    """State of the wisdom ecology"""
    active_psychotechnologies: Dict[PsychotechnologyType, float]
    optimization_modes: Dict[OptimizationMode, float]
    dynamic_constraints: List[Dict]
    emergence_patterns: List[Dict]
    integration_level: float
    meaning_structure: MeaningStructure
    transformation_patterns: List[TransformationPattern]

class EnhancedWisdomEcology:
    """Manages the ecology of psychotechnologies for wisdom cultivation."""
    
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
            integration_level=0.0,
            meaning_structure=MeaningStructure(
                purpose=0.0,
                coherence=0.0,
                significance=0.0,
                connectedness=0.0,
                transcendence=0.0,
                authenticity=0.0
            ),
            transformation_patterns=[]
        )

    def activate_psychotechnology(
        self,
        tech_type: PsychotechnologyType,
        intensity: float = 0.5
    ) -> None:
        """Activate a psychotechnology with cross-framework resonance."""
        self.ecology_state.active_psychotechnologies[tech_type] = min(
            1.0,
            self.ecology_state.active_psychotechnologies[tech_type] + intensity
        )
        self._process_emergence()
        self._update_integration()
        self._update_meaning_structure()
        self._process_transformation()

    def engage_optimization(
        self,
        mode: OptimizationMode,
        intensity: float = 0.5
    ) -> None:
        """Engage an optimization mode with dynamic constraints."""
        self.ecology_state.optimization_modes[mode] = min(
            1.0,
            self.ecology_state.optimization_modes[mode] + intensity
        )
        self._update_constraints()
        self._update_integration()
        self._process_transformation()

    def initiate_transformation(
        self,
        source_state: Dict[str, float],
        target_state: Dict[str, float],
        practices: List[PsychotechnologyType]
    ) -> None:
        """Initiate a transformation pattern."""
        pattern = TransformationPattern(
            source_state=source_state,
            target_state=target_state,
            bridge_practices=practices,
            emergence_factors=[],
            integration_level=0.0
        )
        self.ecology_state.transformation_patterns.append(pattern)
        self._process_transformation()

    def _update_meaning_structure(self) -> None:
        """Update the meaning structure based on active components."""
        # Update purpose through aspiration and gnosis
        self.ecology_state.meaning_structure.purpose = min(
            1.0,
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.ASPIRATION] +
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.GNOSIS] +
            0.2 * self.ecology_state.integration_level
        )
        
        # Update coherence through understanding and insight
        self.ecology_state.meaning_structure.coherence = min(
            1.0,
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.UNDERSTANDING] +
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.INSIGHT] +
            0.2 * self.ecology_state.optimization_modes[OptimizationMode.HORIZONTAL]
        )
        
        # Update significance through presence and becoming
        self.ecology_state.meaning_structure.significance = min(
            1.0,
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.PRESENCE] +
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.BECOMING] +
            0.2 * self.ecology_state.optimization_modes[OptimizationMode.PARTICIPATORY]
        )
        
        # Update connectedness through internalization
        self.ecology_state.meaning_structure.connectedness = min(
            1.0,
            0.6 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.INTERNALIZATION] +
            0.4 * self.ecology_state.optimization_modes[OptimizationMode.DIALOGICAL]
        )
        
        # Update transcendence through courage and presence
        self.ecology_state.meaning_structure.transcendence = min(
            1.0,
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.COURAGE] +
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.PRESENCE] +
            0.2 * self.ecology_state.optimization_modes[OptimizationMode.TRANSFORMATIVE]
        )
        
        # Update authenticity through becoming and presence
        self.ecology_state.meaning_structure.authenticity = min(
            1.0,
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.BECOMING] +
            0.4 * self.ecology_state.active_psychotechnologies[PsychotechnologyType.PRESENCE] +
            0.2 * self.ecology_state.integration_level
        )

    def _process_transformation(self) -> None:
        """Process transformation patterns in the ecology."""
        for pattern in self.ecology_state.transformation_patterns:
            # Calculate practice effectiveness
            practice_factor = sum(
                self.ecology_state.active_psychotechnologies[practice]
                for practice in pattern.bridge_practices
            ) / len(pattern.bridge_practices)
            
            # Calculate emergence strength
            emergence_factor = len(pattern.emergence_factors) / 10.0
            
            # Update integration level
            pattern.integration_level = min(
                1.0,
                0.4 * practice_factor +
                0.3 * emergence_factor +
                0.3 * self.ecology_state.optimization_modes[OptimizationMode.TRANSFORMATIVE]
            )
            
            # Check for new emergent factors
            if pattern.integration_level > 0.7:
                if "coherence" not in pattern.emergence_factors:
                    pattern.emergence_factors.append("coherence")
                if "transcendence" not in pattern.emergence_factors:
                    pattern.emergence_factors.append("transcendence")

    def evaluate_ecology(self) -> Dict[str, float]:
        """Evaluate the current state of the wisdom ecology."""
        active_techs = sum(
            1.0 for level in self.ecology_state.active_psychotechnologies.values()
            if level > 0.5
        ) / len(PsychotechnologyType)
        
        constraints = len(self.ecology_state.dynamic_constraints) / 10.0
        emergence = len(self.ecology_state.emergence_patterns) / 5.0
        
        meaning_factor = (
            self.ecology_state.meaning_structure.purpose +
            self.ecology_state.meaning_structure.coherence +
            self.ecology_state.meaning_structure.significance +
            self.ecology_state.meaning_structure.connectedness +
            self.ecology_state.meaning_structure.transcendence +
            self.ecology_state.meaning_structure.authenticity
        ) / 6.0
        
        transformation_factor = sum(
            pattern.integration_level
            for pattern in self.ecology_state.transformation_patterns
        ) / max(len(self.ecology_state.transformation_patterns), 1)
        
        return {
            "active_technologies": active_techs,
            "dynamic_constraints": constraints,
            "emergence_patterns": emergence,
            "integration_level": self.ecology_state.integration_level,
            "meaning_factor": meaning_factor,
            "transformation_factor": transformation_factor,
            "overall_ecology": (
                0.2 * active_techs +
                0.15 * constraints +
                0.15 * emergence +
                0.2 * self.ecology_state.integration_level +
                0.15 * meaning_factor +
                0.15 * transformation_factor
            )
        }

    def get_recommendations(self) -> List[str]:
        """Get recommendations for enhancing the wisdom ecology."""
        recommendations = []
        
        # Check for weak psychotechnologies
        weak_techs = [
            tech for tech, level in self.ecology_state.active_psychotechnologies.items()
            if level < 0.3
        ]
        if weak_techs:
            recommendations.append(
                f"Strengthen psychotechnologies: {', '.join(t.value for t in weak_techs)}"
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
            
        # Check meaning structure
        meaning_attrs = {
            "purpose": self.ecology_state.meaning_structure.purpose,
            "coherence": self.ecology_state.meaning_structure.coherence,
            "significance": self.ecology_state.meaning_structure.significance,
            "connectedness": self.ecology_state.meaning_structure.connectedness,
            "transcendence": self.ecology_state.meaning_structure.transcendence,
            "authenticity": self.ecology_state.meaning_structure.authenticity
        }
        weak_meaning = [
            attr for attr, level in meaning_attrs.items()
            if level < 0.4
        ]
        if weak_meaning:
            recommendations.append(
                f"Develop meaning dimensions: {', '.join(weak_meaning)}"
            )
            
        # Check transformation patterns
        if not self.ecology_state.transformation_patterns:
            recommendations.append(
                "Initiate transformation patterns through practice integration"
            )
        elif any(pattern.integration_level < 0.4 for pattern in self.ecology_state.transformation_patterns):
            recommendations.append(
                "Strengthen existing transformation patterns through practice"
            )
            
        return recommendations 