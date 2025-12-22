from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

class LandscapeType(Enum):
    SALIENCE = "salience"       # What stands out and how
    PRESENCE = "presence"       # Affordances and co-identification
    DEPTH = "depth"            # Causal patterns and understanding
    SIGNIFICANCE = "significance" # Integration of all landscapes

@dataclass
class Feature:
    """Represents a feature that has been picked out from the environment"""
    name: str
    salience: float  # How much it stands out
    foregrounded: bool  # Whether it's in foreground or background
    configuration: Optional[str]  # How it's configured with other features

@dataclass
class Affordance:
    """Represents an action possibility in agent-arena relationship"""
    name: str
    agent_aspect: str  # What aspect of agent is involved
    arena_aspect: str  # What aspect of environment is involved
    strength: float   # How strong the affordance is
    co_identification_level: float  # How well agent-arena are matched

@dataclass
class CausalPattern:
    """Represents a causal relationship discovered through interaction"""
    cause: str
    effect: str
    reliability: float  # How reliable the pattern is
    understanding_depth: float  # How deeply it's understood

@dataclass
class ConsciousnessLandscape:
    """Represents the current state of all landscapes"""
    features: List[Feature]
    affordances: List[Affordance]
    causal_patterns: List[CausalPattern]
    integration_level: float  # How well the landscapes are integrated
    transformation_potential: float  # Potential for developmental transformation

class ConsciousnessManager:
    """Manages consciousness landscapes and their transformations"""
    
    def __init__(self):
        self.current_landscape = ConsciousnessLandscape(
            features=[],
            affordances=[],
            causal_patterns=[],
            integration_level=0.0,
            transformation_potential=0.0
        )
        self.landscape_history: List[ConsciousnessLandscape] = []
    
    def add_feature(self, name: str, salience: float, foregrounded: bool = False) -> None:
        """Adds a new feature to the salience landscape"""
        feature = Feature(
            name=name,
            salience=salience,
            foregrounded=foregrounded,
            configuration=None
        )
        self.current_landscape.features.append(feature)
        self._update_integration()
    
    def create_affordance(self, name: str, agent: str, arena: str, strength: float) -> None:
        """Creates a new affordance in the presence landscape"""
        affordance = Affordance(
            name=name,
            agent_aspect=agent,
            arena_aspect=arena,
            strength=strength,
            co_identification_level=min(strength, 
                self._calculate_co_identification(agent, arena))
        )
        self.current_landscape.affordances.append(affordance)
        self._update_integration()
    
    def discover_causal_pattern(self, cause: str, effect: str, reliability: float) -> None:
        """Adds a new causal pattern to the depth landscape"""
        pattern = CausalPattern(
            cause=cause,
            effect=effect,
            reliability=reliability,
            understanding_depth=self._calculate_understanding_depth(cause, effect)
        )
        self.current_landscape.causal_patterns.append(pattern)
        self._update_integration()
    
    def _calculate_co_identification(self, agent: str, arena: str) -> float:
        """Calculates how well agent and arena aspects are co-identified"""
        # Calculate co-identification based on:
        # 1. Semantic similarity of agent and arena features
        # 2. Number of shared affordances
        # 3. Historical interaction patterns
        
        # Count matching features
        matching_features = 0
        total_features = 0
        
        for feature in self.current_landscape.features:
            if agent.lower() in feature.name.lower() or arena.lower() in feature.name.lower():
                total_features += 1
                # Check if there's a strong salience (indicates good co-identification)
                if feature.salience > 0.7:
                    matching_features += 1
        
        # Count shared affordances
        shared_affordances = 0
        for affordance in self.current_landscape.affordances:
            if (agent.lower() in affordance.agent_aspect.lower() and 
                arena.lower() in affordance.arena_aspect.lower()):
                shared_affordances += 1
        
        # Calculate co-identification score
        if total_features == 0:
            feature_score = 0.5  # Neutral if no features
        else:
            feature_score = matching_features / total_features
        
        affordance_score = min(shared_affordances / 5.0, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        co_identification = (0.6 * feature_score) + (0.4 * affordance_score)
        
        return co_identification
    
    def _calculate_understanding_depth(self, cause: str, effect: str) -> float:
        """Calculates how deeply a causal pattern is understood"""
        # Calculate understanding depth based on:
        # 1. Number of related causal patterns
        # 2. Integration with feature and affordance landscapes
        # 3. Reliability of the pattern
        
        # Count related patterns
        related_patterns = 0
        total_reliability = 0.0
        
        for pattern in self.current_landscape.causal_patterns:
            # Check if patterns share cause or effect
            if (cause.lower() in pattern.cause.lower() or 
                effect.lower() in pattern.effect.lower() or
                pattern.cause.lower() in cause.lower() or
                pattern.effect.lower() in effect.lower()):
                related_patterns += 1
                total_reliability += pattern.reliability
        
        # Calculate pattern network depth
        if related_patterns == 0:
            pattern_depth = 0.3  # Low depth if isolated
        else:
            avg_reliability = total_reliability / related_patterns
            pattern_depth = min(related_patterns / 10.0, 0.8) * avg_reliability
        
        # Check integration with features
        feature_integration = 0.0
        for feature in self.current_landscape.features:
            if (cause.lower() in feature.name.lower() or 
                effect.lower() in feature.name.lower()):
                feature_integration += feature.salience
        
        feature_integration = min(feature_integration / 2.0, 0.3)
        
        # Check integration with affordances
        affordance_integration = 0.0
        for affordance in self.current_landscape.affordances:
            if (cause.lower() in affordance.name.lower() or 
                effect.lower() in affordance.name.lower()):
                affordance_integration += affordance.strength
        
        affordance_integration = min(affordance_integration / 2.0, 0.3)
        
        # Combine all factors
        understanding_depth = (
            0.5 * pattern_depth + 
            0.25 * feature_integration + 
            0.25 * affordance_integration
        )
        
        return min(understanding_depth, 1.0)
    
    def _update_integration(self) -> None:
        """Updates the integration level across landscapes"""
        feature_coherence = self._calculate_feature_coherence()
        affordance_coherence = self._calculate_affordance_coherence()
        causal_coherence = self._calculate_causal_coherence()
        
        self.current_landscape.integration_level = (
            feature_coherence + affordance_coherence + causal_coherence
        ) / 3.0
        
        self.current_landscape.transformation_potential = (
            self.current_landscape.integration_level * 
            (1 - self.current_landscape.integration_level)  # Peak at medium integration
        )
        
        # Record current state
        self.landscape_history.append(ConsciousnessLandscape(
            **vars(self.current_landscape)
        ))
    
    def _calculate_feature_coherence(self) -> float:
        """Calculates how coherently features are organized"""
        if not self.current_landscape.features:
            return 0.0
        return sum(f.salience for f in self.current_landscape.features) / len(
            self.current_landscape.features)
    
    def _calculate_affordance_coherence(self) -> float:
        """Calculates how well affordances form a coherent network"""
        if not self.current_landscape.affordances:
            return 0.0
        return sum(a.co_identification_level for a in self.current_landscape.affordances) / len(
            self.current_landscape.affordances)
    
    def _calculate_causal_coherence(self) -> float:
        """Calculates how well causal patterns are understood"""
        if not self.current_landscape.causal_patterns:
            return 0.0
        return sum(p.understanding_depth for p in self.current_landscape.causal_patterns) / len(
            self.current_landscape.causal_patterns)
    
    def evaluate_consciousness(self) -> Dict[str, float]:
        """Evaluates the current state of consciousness"""
        return {
            "salience_coherence": self._calculate_feature_coherence(),
            "presence_coherence": self._calculate_affordance_coherence(),
            "depth_coherence": self._calculate_causal_coherence(),
            "overall_integration": self.current_landscape.integration_level,
            "transformation_potential": self.current_landscape.transformation_potential
        } 