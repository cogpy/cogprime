from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

from .continuity_hypothesis import ContinuityManager, OptimalGripDomain, ProcessingState
from .plausibility_framework import PlausibilityManager, EvidenceType, ApplicationDomain
from .parasitic_processing import ParasiticProcessingManager, PathComponent

class WisdomAspect(Enum):
    PROPOSITIONAL = "propositional"   # Knowledge that (facts, concepts)
    PROCEDURAL = "procedural"         # Knowledge how (skills, techniques)
    PERSPECTIVAL = "perspectival"     # Seeing through different perspectives
    PARTICIPATORY = "participatory"    # Direct engagement with reality
    TRANSFORMATIVE = "transformative"  # Self-transcendence through integration

@dataclass
class WisdomState:
    """Represents the current state of wisdom development"""
    propositional_knowledge: float  # Grasp of factual knowledge
    procedural_mastery: float      # Skill in applying wisdom
    perspectival_insight: float    # Ability to see through perspectives
    participatory_knowing: float   # Depth of engagement with reality
    transformative_depth: float    # Degree of self-transcendence

@dataclass
class IntegrationMetrics:
    """Tracks how well the different aspects are integrated"""
    grip_plausibility: float      # How optimal grip affects plausibility
    plausibility_parasitic: float # How plausibility affects parasitic processing
    parasitic_grip: float         # How parasitic processing affects grip
    overall_coherence: float      # Overall integration level

class WisdomIntegrationManager:
    """Manages the integration of different wisdom-related processes"""
    
    def __init__(self):
        self.continuity_manager = ContinuityManager()
        self.plausibility_manager = PlausibilityManager()
        self.parasitic_manager = ParasiticProcessingManager()
        
        self.wisdom_state = WisdomState(
            propositional_knowledge=0.0,
            procedural_mastery=0.0,
            perspectival_insight=0.0,
            participatory_knowing=0.0,
            transformative_depth=0.0
        )
        
        self.integration_metrics = IntegrationMetrics(
            grip_plausibility=0.0,
            plausibility_parasitic=0.0,
            parasitic_grip=0.0,
            overall_coherence=0.0
        )
        
        self.state_history: List[Tuple[WisdomState, IntegrationMetrics]] = []

    def update_from_optimal_grip(self) -> None:
        """Updates system based on current optimal grip state"""
        continuity_state = self.continuity_manager.evaluate_continuity()
        
        # High grip balance increases plausibility
        if continuity_state["average_grip_balance"] > 0.7:
            self.plausibility_manager.add_application_domain(
                name="optimal_grip",
                relevance=continuity_state["average_grip_balance"],
                effectiveness=continuity_state["overall_adaptation"],
                adaptability=continuity_state["transformation_level"]
            )
        
        # Good optimal grip reduces parasitic processing
        if continuity_state["overall_adaptation"] > 0.7:
            self.parasitic_manager.parasitic_state.cognitive_flexibility += 0.1
            self.parasitic_manager.agent_arena.agent_flexibility += 0.1
        
        self._update_integration_metrics()

    def update_from_plausibility(self) -> None:
        """Updates system based on current plausibility assessment"""
        plausibility_state = self.plausibility_manager.evaluate_plausibility()
        
        # High plausibility enhances optimal grip
        if plausibility_state["overall_plausibility"] > 0.7:
            for domain in OptimalGripDomain:
                self.continuity_manager.adjust_grip(
                    domain=domain,
                    gestalt_emphasis=plausibility_state["elegance"],
                    feature_emphasis=plausibility_state["convergence"]
                )
        
        # High plausibility counters parasitic processing
        if plausibility_state["trustworthiness"] > 0.7:
            self.parasitic_manager.parasitic_state.event_interpretation *= 0.9
            self.parasitic_manager.parasitic_state.probability_assessment *= 0.9
        
        self._update_integration_metrics()

    def update_from_parasitic(self) -> None:
        """Updates system based on current parasitic processing state"""
        parasitic_state = self.parasitic_manager.evaluate_state()
        
        # High parasitic intensity disrupts optimal grip
        if parasitic_state["parasitic_intensity"] > 0.7:
            for domain in OptimalGripDomain:
                self.continuity_manager.adjust_grip(
                    domain=domain,
                    gestalt_emphasis=0.3,  # Reduced clarity
                    feature_emphasis=0.3   # Reduced detail
                )
        
        # High cognitive health enhances plausibility
        if parasitic_state["cognitive_health"] > 0.7:
            self.plausibility_manager.add_evidence(
                type=EvidenceType.PRACTICAL,
                strength=parasitic_state["cognitive_health"],
                independence=1.0 - parasitic_state["parasitic_intensity"],
                reliability=parasitic_state["transformation_potential"],
                scope=["cognitive", "existential"]
            )
        
        self._update_integration_metrics()

    def _update_integration_metrics(self) -> None:
        """Updates metrics tracking integration between systems"""
        continuity_state = self.continuity_manager.evaluate_continuity()
        plausibility_state = self.plausibility_manager.evaluate_plausibility()
        parasitic_state = self.parasitic_manager.evaluate_state()
        
        # Calculate integration between systems
        self.integration_metrics.grip_plausibility = (
            continuity_state["average_grip_balance"] * 
            plausibility_state["overall_plausibility"]
        )
        
        self.integration_metrics.plausibility_parasitic = (
            plausibility_state["trustworthiness"] * 
            (1.0 - parasitic_state["parasitic_intensity"])
        )
        
        self.integration_metrics.parasitic_grip = (
            parasitic_state["cognitive_health"] *
            continuity_state["overall_adaptation"]
        )
        
        # Overall coherence considers all interactions
        self.integration_metrics.overall_coherence = (
            self.integration_metrics.grip_plausibility *
            self.integration_metrics.plausibility_parasitic *
            self.integration_metrics.parasitic_grip
        ) ** (1/3)  # Geometric mean
        
        # Update wisdom state based on integration
        self.wisdom_state.propositional_knowledge = (
            plausibility_state["power"] *
            (1.0 - parasitic_state["parasitic_intensity"])
        )
        
        self.wisdom_state.procedural_mastery = (
            parasitic_state["cognitive_health"] *
            plausibility_state["fluency"]
        )
        
        self.wisdom_state.perspectival_insight = (
            continuity_state["insight_readiness"] *
            plausibility_state["elegance"]
        )
        
        self.wisdom_state.participatory_knowing = (
            plausibility_state["convergence"] *
            continuity_state["flow_intensity"]
        )
        
        self.wisdom_state.transformative_depth = (
            continuity_state["transformation_level"] *
            parasitic_state["transformation_potential"]
        )
        
        # Record state
        self.state_history.append((
            WisdomState(**vars(self.wisdom_state)),
            IntegrationMetrics(**vars(self.integration_metrics))
        ))

    def evaluate_wisdom(self) -> Dict[str, float]:
        """Evaluates the current state of wisdom development"""
        return {
            "propositional_knowledge": self.wisdom_state.propositional_knowledge,
            "procedural_mastery": self.wisdom_state.procedural_mastery,
            "perspectival_insight": self.wisdom_state.perspectival_insight,
            "participatory_knowing": self.wisdom_state.participatory_knowing,
            "transformative_depth": self.wisdom_state.transformative_depth,
            "grip_plausibility": self.integration_metrics.grip_plausibility,
            "plausibility_parasitic": self.integration_metrics.plausibility_parasitic,
            "parasitic_grip": self.integration_metrics.parasitic_grip,
            "overall_coherence": self.integration_metrics.overall_coherence
        }

    def get_recommendations(self) -> List[str]:
        """Provides recommendations for wisdom development"""
        recommendations = []
        
        # Check wisdom aspects
        if self.wisdom_state.propositional_knowledge < 0.4:
            recommendations.append("Study core concepts and theoretical frameworks")
        if self.wisdom_state.procedural_mastery < 0.4:
            recommendations.append("Practice applying wisdom techniques in daily life")
        if self.wisdom_state.perspectival_insight < 0.4:
            recommendations.append("Explore different perspectives and worldviews")
        if self.wisdom_state.participatory_knowing < 0.4:
            recommendations.append("Deepen direct engagement through contemplative practices")
        if self.wisdom_state.transformative_depth < 0.4:
            recommendations.append("Integrate insights across ways of knowing")
            
        # Check integration
        if self.integration_metrics.overall_coherence < 0.4:
            recommendations.append("Work on bridging theory and practice across domains")
        
        # Get recommendations from component managers
        recommendations.extend(self.continuity_manager.get_recommendations())
        recommendations.extend(self.plausibility_manager.get_improvement_suggestions())
        recommendations.extend(self.parasitic_manager.get_interventions())
        
        return recommendations 