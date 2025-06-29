from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

class KnowingMode(Enum):
    """Modes of knowing in the Romantic framework"""
    RATIONAL = "rational"  # Logical, mathematical knowing
    IMAGINATIVE = "imaginative"  # Creative, expressive knowing
    PARTICIPATORY = "participatory"  # Direct experiential knowing
    PERSPECTIVAL = "perspectival"  # Subjective viewpoint knowing

class ExpressionMode(Enum):
    """Modes of romantic expression"""
    ARTISTIC = "artistic"  # Aesthetic creation
    POETIC = "poetic"  # Literary expression
    MUSICAL = "musical"  # Musical expression
    MYTHOLOGICAL = "mythological"  # Mythic narrative

class IdeologyType(Enum):
    """Types of pseudo-religious ideology"""
    ROMANTIC = "romantic"  # Original romantic movement
    NATIONALIST = "nationalist"  # National identity based
    TOTALITARIAN = "totalitarian"  # Complete system of meaning
    DECADENT = "decadent"  # Degraded romantic forms

@dataclass
class ImaginativeState:
    """Represents a state of romantic imagination"""
    mode: KnowingMode
    expression: ExpressionMode
    intensity: float  # Degree of imaginative engagement
    world_contact: float  # Degree of contact with reality
    transformative_power: float  # Capacity for transformation

@dataclass
class RomanticResponse:
    """Models the romantic response to rationalism"""
    knowing_modes: List[KnowingMode]
    expression_modes: List[ExpressionMode]
    world_relationship: str
    transformative_potential: float

class RomanticImagination:
    """Models the romantic conception of imagination"""
    
    def __init__(self):
        self.current_state = self._initialize_state()
        self.expressive_modes = self._initialize_modes()
        
    def _initialize_state(self) -> ImaginativeState:
        """Initialize the default imaginative state"""
        return ImaginativeState(
            mode=KnowingMode.IMAGINATIVE,
            expression=ExpressionMode.ARTISTIC,
            intensity=0.5,
            world_contact=0.5,
            transformative_power=0.5
        )
    
    def _initialize_modes(self) -> Dict[ExpressionMode, float]:
        """Initialize the expressive modes with their potencies"""
        return {
            ExpressionMode.ARTISTIC: 0.8,
            ExpressionMode.POETIC: 0.9,
            ExpressionMode.MUSICAL: 0.7,
            ExpressionMode.MYTHOLOGICAL: 1.0
        }
    
    def express(self, mode: ExpressionMode) -> float:
        """Express imagination through a particular mode"""
        potency = self.expressive_modes[mode]
        self.current_state.intensity *= potency
        return self.current_state.intensity
    
    def assess_world_contact(self) -> float:
        """Assess degree of contact with reality"""
        # Romantic paradox: more imaginative = less rational but more "real"
        return self.current_state.world_contact * self.current_state.intensity

class IdeologicalTransformation:
    """Models the transformation of romanticism into ideology"""
    
    def __init__(self):
        self.current_type = IdeologyType.ROMANTIC
        self.transformation_stages = []
        self.decadence_level = 0.0
        
    def transform(self, target_type: IdeologyType) -> bool:
        """Transform romantic impulse into different ideological form"""
        if self._can_transform(target_type):
            self.transformation_stages.append(self.current_type)
            self.current_type = target_type
            self._update_decadence()
            return True
        return False
    
    def _can_transform(self, target_type: IdeologyType) -> bool:
        """Check if transformation to target type is possible"""
        # Prevent regression to original romantic form
        if target_type == IdeologyType.ROMANTIC and len(self.transformation_stages) > 0:
            return False
        return True
    
    def _update_decadence(self):
        """Update decadence level based on transformations"""
        self.decadence_level += 0.2 * len(self.transformation_stages)
        if self.current_type == IdeologyType.DECADENT:
            self.decadence_level = 1.0

class RomanticMovement:
    """Models the overall romantic movement and its transformations"""
    
    def __init__(self):
        self.imagination = RomanticImagination()
        self.ideology = IdeologicalTransformation()
        self.responses = []
        
    def generate_response(self, rational_pressure: float) -> RomanticResponse:
        """Generate a romantic response to rationalist pressure"""
        intensity = 1.0 - rational_pressure  # Inverse relationship
        
        response = RomanticResponse(
            knowing_modes=[KnowingMode.IMAGINATIVE, KnowingMode.PARTICIPATORY],
            expression_modes=[ExpressionMode.POETIC, ExpressionMode.MYTHOLOGICAL],
            world_relationship="direct_contact" if intensity > 0.7 else "mediated",
            transformative_potential=intensity
        )
        
        self.responses.append(response)
        return response
    
    def assess_movement_state(self) -> Dict[str, float]:
        """Assess the current state of the romantic movement"""
        return {
            "imaginative_power": self.imagination.current_state.intensity,
            "world_contact": self.imagination.assess_world_contact(),
            "ideological_decadence": self.ideology.decadence_level,
            "transformative_potential": sum(r.transformative_potential 
                                         for r in self.responses) / len(self.responses)
            if self.responses else 0.0
        } 