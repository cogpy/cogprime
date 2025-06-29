from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

class ProcessingPattern(Enum):
    ADAPTIVE = "adaptive"         # Normal adaptive processing
    PARASITIC = "parasitic"      # Self-destructive processing
    RECIPROCAL = "reciprocal"    # Mutually reinforcing patterns
    TRANSFORMATIVE = "transformative"  # Growth-oriented patterns

class NobleProvocation(Enum):
    DUKKHA_THREAT = "dukkha_threat"  # Life threatened by loss of agency
    ATTACHMENT_CAUSE = "attachment_cause"  # Attachment causes dukkha
    FREEDOM_POSSIBLE = "freedom_possible"  # Agency can be recovered
    EIGHTFOLD_PATH = "eightfold_path"  # Path to transformation

class PathComponent(Enum):
    RIGHT_UNDERSTANDING = "right_understanding"
    RIGHT_THINKING = "right_thinking"
    RIGHT_SPEECH = "right_speech"
    RIGHT_ACTION = "right_action"
    RIGHT_LIVELIHOOD = "right_livelihood"
    RIGHT_MINDFULNESS = "right_mindfulness"
    RIGHT_CONCENTRATION = "right_concentration"

@dataclass
class ParasiticState:
    """Represents the state of parasitic processing"""
    event_interpretation: float  # How negatively events are interpreted
    probability_assessment: float  # Perceived probability of negative events
    anxiety_level: float  # Current anxiety level
    cognitive_flexibility: float  # Remaining flexibility
    problem_solving: float  # Problem-solving capability
    fatalism: float  # Degree of fatalistic thinking

@dataclass
class AgentArenaRelation:
    """Represents the relationship between agent and arena"""
    agent_flexibility: float  # Agent's adaptability
    arena_options: float  # Perceived options in environment
    reciprocal_narrowing: float  # Degree of mutual constraint
    transformation_potential: float  # Potential for positive change

@dataclass
class CognitiveHeuristic:
    """Represents a cognitive shortcut used in processing"""
    name: str
    strength: float  # How strongly it influences processing
    adaptiveness: float  # How adaptive vs maladaptive it is
    domain: str  # Where it operates
    bias_potential: float  # Potential for creating bias

@dataclass
class DynamicalSystem:
    """Represents a self-organizing system of cognitive processes"""
    components: List[str]
    interactions: Dict[str, List[str]]  # How components affect each other
    stability: float  # How stable the system is
    adaptiveness: float  # How adaptive the system is
    transformation_rate: float  # Rate of self-modification

class ParasiticProcessingManager:
    """Manages parasitic processing and transformative practices"""
    
    def __init__(self):
        self.parasitic_state = ParasiticState(
            event_interpretation=0.0,
            probability_assessment=0.0,
            anxiety_level=0.0,
            cognitive_flexibility=1.0,
            problem_solving=1.0,
            fatalism=0.0
        )
        
        self.agent_arena = AgentArenaRelation(
            agent_flexibility=1.0,
            arena_options=1.0,
            reciprocal_narrowing=0.0,
            transformation_potential=1.0
        )
        
        self.active_heuristics: List[CognitiveHeuristic] = []
        self.parasitic_system = DynamicalSystem(
            components=[],
            interactions={},
            stability=0.0,
            adaptiveness=1.0,
            transformation_rate=0.0
        )
        self.counteractive_system = DynamicalSystem(
            components=list(PathComponent),
            interactions={},
            stability=0.0,
            adaptiveness=1.0,
            transformation_rate=0.0
        )
        
        self.active_components: Set[PathComponent] = set()
        self.state_history: List[Tuple[ParasiticState, AgentArenaRelation]] = []

    def process_event(self, 
                     negative_interpretation: float,
                     anxiety_trigger: bool) -> None:
        """Processes an event and updates parasitic state"""
        # Update event interpretation
        self.parasitic_state.event_interpretation = max(0.0, min(1.0,
            self.parasitic_state.event_interpretation + negative_interpretation))
        
        # Update probability assessment based on availability/representativeness
        if self.parasitic_state.event_interpretation > 0.6:
            self.parasitic_state.probability_assessment += 0.1
            
        # Anxiety affects cognitive flexibility
        if anxiety_trigger:
            self.parasitic_state.anxiety_level += 0.2
            self.parasitic_state.cognitive_flexibility *= 0.8
            
        # Reduced flexibility impacts problem solving
        self.parasitic_state.problem_solving = (
            self.parasitic_state.cognitive_flexibility * 
            (1.0 - self.parasitic_state.anxiety_level)
        )
        
        # High anxiety and low problem solving lead to fatalism
        if self.parasitic_state.anxiety_level > 0.7 and \
           self.parasitic_state.problem_solving < 0.3:
            self.parasitic_state.fatalism += 0.1
            
        self._update_agent_arena()
        self._update_dynamical_systems()
        
        # Record state
        self.state_history.append((
            ParasiticState(**vars(self.parasitic_state)),
            AgentArenaRelation(**vars(self.agent_arena))
        ))

    def add_heuristic(self,
                     name: str,
                     strength: float,
                     adaptiveness: float,
                     domain: str) -> None:
        """Adds a cognitive heuristic to active processing"""
        heuristic = CognitiveHeuristic(
            name=name,
            strength=strength,
            adaptiveness=adaptiveness,
            domain=domain,
            bias_potential=strength * (1.0 - adaptiveness)
        )
        self.active_heuristics.append(heuristic)
        self._update_dynamical_systems()

    def _update_agent_arena(self) -> None:
        """Updates the agent-arena relationship"""
        # Agent flexibility decreases with parasitic processing
        self.agent_arena.agent_flexibility = (
            1.0 - self.parasitic_state.anxiety_level
        )
        
        # Perceived options decrease with fatalism
        self.agent_arena.arena_options = (
            1.0 - self.parasitic_state.fatalism
        )
        
        # Calculate reciprocal narrowing
        self.agent_arena.reciprocal_narrowing = (
            1.0 - (self.agent_arena.agent_flexibility * 
                   self.agent_arena.arena_options)
        )
        
        # Calculate transformation potential
        self.agent_arena.transformation_potential = (
            self.parasitic_state.cognitive_flexibility *
            (1.0 - self.agent_arena.reciprocal_narrowing)
        )

    def _update_dynamical_systems(self) -> None:
        """Updates both parasitic and counteractive dynamical systems"""
        # Update parasitic system
        self.parasitic_system.components = [h.name for h in self.active_heuristics]
        self.parasitic_system.stability = sum(
            h.strength for h in self.active_heuristics
        ) / len(self.active_heuristics) if self.active_heuristics else 0.0
        self.parasitic_system.adaptiveness = sum(
            h.adaptiveness for h in self.active_heuristics
        ) / len(self.active_heuristics) if self.active_heuristics else 1.0
        
        # Update counteractive system
        active_count = len(self.active_components)
        if active_count > 0:
            self.counteractive_system.stability = active_count / len(PathComponent)
            self.counteractive_system.transformation_rate = (
                self.counteractive_system.stability * 
                self.agent_arena.transformation_potential
            )

    def apply_path_component(self, component: PathComponent) -> None:
        """Applies an eightfold path component to counter parasitic processing"""
        self.active_components.add(component)
        
        # Different components affect different aspects
        if component in {PathComponent.RIGHT_UNDERSTANDING, 
                        PathComponent.RIGHT_THINKING}:
            self.parasitic_state.cognitive_flexibility += 0.1
            self.parasitic_state.event_interpretation *= 0.9
            
        elif component in {PathComponent.RIGHT_SPEECH, 
                          PathComponent.RIGHT_ACTION,
                          PathComponent.RIGHT_LIVELIHOOD}:
            self.parasitic_state.anxiety_level *= 0.9
            self.agent_arena.agent_flexibility += 0.1
            
        elif component in {PathComponent.RIGHT_MINDFULNESS,
                          PathComponent.RIGHT_CONCENTRATION}:
            self.parasitic_state.fatalism *= 0.9
            self.agent_arena.arena_options += 0.1
            
        self._update_agent_arena()
        self._update_dynamical_systems()

    def evaluate_state(self) -> Dict[str, float]:
        """Evaluates the current state of processing"""
        return {
            "parasitic_intensity": (
                self.parasitic_state.anxiety_level +
                self.parasitic_state.fatalism
            ) / 2.0,
            "cognitive_health": (
                self.parasitic_state.cognitive_flexibility +
                self.parasitic_state.problem_solving
            ) / 2.0,
            "reciprocal_narrowing": self.agent_arena.reciprocal_narrowing,
            "transformation_potential": self.agent_arena.transformation_potential,
            "path_coverage": len(self.active_components) / len(PathComponent),
            "parasitic_system_strength": self.parasitic_system.stability,
            "counteractive_system_strength": self.counteractive_system.stability
        }

    def get_interventions(self) -> List[str]:
        """Suggests interventions based on current state"""
        interventions = []
        
        if self.parasitic_state.anxiety_level > 0.6:
            interventions.append("Practice anxiety reduction techniques")
        if self.parasitic_state.cognitive_flexibility < 0.4:
            interventions.append("Engage in cognitive flexibility exercises")
        if self.parasitic_state.fatalism > 0.6:
            interventions.append("Challenge fatalistic thinking patterns")
            
        # Suggest missing path components
        missing_components = set(PathComponent) - self.active_components
        if missing_components:
            interventions.append(
                f"Incorporate {', '.join(c.value)} practices"
            )
            
        return interventions 