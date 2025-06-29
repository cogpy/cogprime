from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

class LoveType(Enum):
    EROS = auto()  # Love of becoming one with something
    PHILIA = auto()  # Love born out of cooperation
    AGAPE = auto()  # Love of creation and transformation

@dataclass
class TransformativeState:
    """Represents the state of transformative experience through love."""
    love_type: LoveType
    is_egocentric: bool
    is_creative: bool
    is_sacrificial: bool
    transforms_other: bool
    transforms_self: bool

@dataclass
class AgapicTransformation:
    """Models the transformative process of Agapic love."""
    from_state: str  # Initial state of being
    to_state: str  # Transformed state of being
    is_metanoia: bool  # Whether radical reorientation occurred
    is_kairos: bool  # Whether it's a turning point
    sacrificial_element: Optional[str]  # Nature of sacrifice involved

class AgapeManager:
    """Manages the dynamics of Agapic love and transformation."""
    
    def __init__(self):
        self.transformations: List[AgapicTransformation] = []
        self.current_state = TransformativeState(
            love_type=LoveType.EROS,
            is_egocentric=True,
            is_creative=False,
            is_sacrificial=False,
            transforms_other=False,
            transforms_self=False
        )
    
    def initiate_agapic_transformation(self, from_state: str, to_state: str) -> AgapicTransformation:
        """Initiates a new Agapic transformation process."""
        transformation = AgapicTransformation(
            from_state=from_state,
            to_state=to_state,
            is_metanoia=True,
            is_kairos=True,
            sacrificial_element="Giving before receiving"
        )
        self.transformations.append(transformation)
        return transformation
    
    def evolve_love_state(self, new_love_type: LoveType) -> TransformativeState:
        """Evolves the current state based on the type of love."""
        is_agape = new_love_type == LoveType.AGAPE
        self.current_state = TransformativeState(
            love_type=new_love_type,
            is_egocentric=not is_agape,
            is_creative=is_agape,
            is_sacrificial=is_agape,
            transforms_other=is_agape,
            transforms_self=is_agape
        )
        return self.current_state
    
    def apply_forgiveness(self) -> bool:
        """Applies the concept of forgiveness as a form of Agapic love."""
        if self.current_state.love_type != LoveType.AGAPE:
            self.evolve_love_state(LoveType.AGAPE)
        return self.current_state.is_sacrificial and self.current_state.transforms_other
    
    def assess_transformation(self, transformation: AgapicTransformation) -> bool:
        """Assesses whether a transformation embodies true Agapic principles."""
        return (
            transformation.is_metanoia 
            and transformation.is_kairos 
            and transformation.sacrificial_element is not None
        ) 