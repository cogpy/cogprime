from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

class ScalingDirection(Enum):
    UP = "up"      # Moving toward broader patterns/gestalts
    DOWN = "down"  # Moving toward features/components
    IN = "in"      # Moving toward transparency/looking through
    OUT = "out"    # Moving toward opacity/looking at

class MysticalState(Enum):
    PURE_CONSCIOUSNESS = "pure_consciousness"  # PCE - complete stepping back
    RESONANT_ATONEMENT = "resonant_atonement" # Complete scaling up/flow
    NON_DUAL = "non_dual"                     # Integration of both

@dataclass
class ScalingDynamics:
    """Represents the current state of attentional scaling"""
    up_down_position: float  # -1 (complete down) to 1 (complete up)
    in_out_position: float   # -1 (complete in) to 1 (complete out)
    scaling_momentum: Dict[ScalingDirection, float]  # Current movement in each direction
    oscillation_frequency: float  # For practices like breath-synchronized scaling

@dataclass 
class MysticalExperience:
    """Represents a mystical experience and its characteristics"""
    state: MysticalState
    intensity: float  # 0-1 scale
    duration: float   # In seconds
    transformative_potential: float  # Estimated impact on insight/meaning
    integration_level: float  # How well integrated with normal experience

class EnhancedMindfulnessManager:
    """Manages mindfulness practices with advanced scaling and mystical states"""
    
    def __init__(self):
        self.scaling_dynamics = ScalingDynamics(
            up_down_position=0.0,
            in_out_position=0.0,
            scaling_momentum={d: 0.0 for d in ScalingDirection},
            oscillation_frequency=0.0
        )
        self.current_mystical_state: Optional[MysticalExperience] = None
        self.practice_history: List[Tuple[ScalingDynamics, Optional[MysticalExperience]]] = []
    
    def scale_attention(self, direction: ScalingDirection, magnitude: float) -> None:
        """Scales attention in the specified direction with given magnitude"""
        if direction in {ScalingDirection.UP, ScalingDirection.DOWN}:
            self.scaling_dynamics.up_down_position += (
                magnitude if direction == ScalingDirection.UP else -magnitude
            )
            self.scaling_dynamics.up_down_position = max(-1.0, min(1.0, 
                self.scaling_dynamics.up_down_position))
        else:
            self.scaling_dynamics.in_out_position += (
                magnitude if direction == ScalingDirection.OUT else -magnitude
            )
            self.scaling_dynamics.in_out_position = max(-1.0, min(1.0, 
                self.scaling_dynamics.in_out_position))
        
        self._update_mystical_state()
    
    def synchronize_with_breath(self, frequency: float) -> None:
        """Synchronizes scaling with breath for advanced practices"""
        self.scaling_dynamics.oscillation_frequency = frequency
        # Implement breath-synchronized scaling logic
    
    def _update_mystical_state(self) -> None:
        """Updates mystical state based on current scaling dynamics"""
        # Pure Consciousness Event check
        if self.scaling_dynamics.in_out_position <= -0.9:
            self.current_mystical_state = MysticalExperience(
                state=MysticalState.PURE_CONSCIOUSNESS,
                intensity=-self.scaling_dynamics.in_out_position,
                duration=0.0,  # Will be updated as state persists
                transformative_potential=0.8,
                integration_level=0.3
            )
        # Resonant Atonement check
        elif self.scaling_dynamics.up_down_position >= 0.9:
            self.current_mystical_state = MysticalExperience(
                state=MysticalState.RESONANT_ATONEMENT,
                intensity=self.scaling_dynamics.up_down_position,
                duration=0.0,
                transformative_potential=0.7,
                integration_level=0.4
            )
        # Non-Dual State check (requires balance of both dimensions)
        elif abs(self.scaling_dynamics.up_down_position) >= 0.7 and \
             abs(self.scaling_dynamics.in_out_position) >= 0.7:
            self.current_mystical_state = MysticalExperience(
                state=MysticalState.NON_DUAL,
                intensity=min(abs(self.scaling_dynamics.up_down_position),
                            abs(self.scaling_dynamics.in_out_position)),
                duration=0.0,
                transformative_potential=0.9,
                integration_level=0.2
            )
        else:
            self.current_mystical_state = None
        
        # Record state for analysis
        self.practice_history.append((
            ScalingDynamics(**vars(self.scaling_dynamics)),
            self.current_mystical_state
        ))
    
    def evaluate_practice(self) -> Dict[str, float]:
        """Evaluates the current state of practice"""
        return {
            "scaling_balance": 1.0 - abs(self.scaling_dynamics.up_down_position),
            "transparency_opacity_balance": 1.0 - abs(self.scaling_dynamics.in_out_position),
            "mystical_potential": self._calculate_mystical_potential(),
            "transformative_capacity": self._calculate_transformative_capacity()
        }
    
    def _calculate_mystical_potential(self) -> float:
        """Calculates potential for mystical experience"""
        if not self.current_mystical_state:
            return max(
                abs(self.scaling_dynamics.up_down_position),
                abs(self.scaling_dynamics.in_out_position)
            )
        return self.current_mystical_state.intensity
    
    def _calculate_transformative_capacity(self) -> float:
        """Calculates capacity for transformative insight"""
        if not self.current_mystical_state:
            return 0.0
        return (self.current_mystical_state.transformative_potential * 
                self.current_mystical_state.integration_level) 