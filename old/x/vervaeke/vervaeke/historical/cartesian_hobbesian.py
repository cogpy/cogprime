from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

class PropertyType(Enum):
    """Types of properties in the Cartesian-Hobbesian framework"""
    PRIMARY = "primary"  # Mathematically measurable properties
    SECONDARY = "secondary"  # Experiential qualities (qualia)

class SubstanceType(Enum):
    """Types of substances in the Cartesian framework"""
    MATERIAL = "material"  # Extended in space and time
    MENTAL = "mental"  # Thinking substance

class RationalityAspect(Enum):
    """Core aspects of rationality according to Descartes"""
    PURPOSE = "purpose"  # Goal-directed behavior
    MEANING = "meaning"  # Semantic content
    NORMATIVITY = "normativity"  # Standards of correctness
    TRUTH = "truth"  # Pursuit of truth

@dataclass
class Property:
    """A property that can be attributed to mind or matter"""
    name: str
    type: PropertyType
    measurable: bool
    experiential: bool
    description: str

@dataclass
class MentalState:
    """Represents a state of mind with Cartesian characteristics"""
    consciousness: float  # Degree of conscious awareness
    rationality: List[RationalityAspect]  # Active rational aspects
    qualia: List[str]  # Experiential qualities
    certainty: float  # Degree of Cartesian certainty

@dataclass
class MaterialState:
    """Represents a state of matter with mechanical properties"""
    extension: Dict[str, float]  # Spatial dimensions
    motion: Dict[str, float]  # Mechanical properties
    measurable_properties: List[Property]
    causal_relations: List[str]

class CartesianMindMatterDivide:
    """Models the fundamental divide between mind and matter in Cartesian philosophy"""
    
    def __init__(self):
        self.mental_properties = self._initialize_mental_properties()
        self.material_properties = self._initialize_material_properties()
        self.interaction_problems = []

    def _initialize_mental_properties(self) -> List[Property]:
        """Initialize core mental properties"""
        return [
            Property("consciousness", PropertyType.SECONDARY, False, True, "Self-awareness"),
            Property("rationality", PropertyType.SECONDARY, False, True, "Capacity for reason"),
            Property("purpose", PropertyType.SECONDARY, False, True, "Goal-directedness"),
            Property("meaning", PropertyType.SECONDARY, False, True, "Semantic content")
        ]

    def _initialize_material_properties(self) -> List[Property]:
        """Initialize core material properties"""
        return [
            Property("extension", PropertyType.PRIMARY, True, False, "Spatial dimensions"),
            Property("motion", PropertyType.PRIMARY, True, False, "Mechanical movement"),
            Property("mass", PropertyType.PRIMARY, True, False, "Material quantity"),
            Property("force", PropertyType.PRIMARY, True, False, "Mechanical interaction")
        ]

    def assess_interaction_possibility(self, mental: MentalState, material: MaterialState) -> bool:
        """Assess whether mind-matter interaction is possible given their states"""
        # According to Descartes, interaction should be impossible due to property mismatch
        has_shared_properties = any(
            m.type == p.type for m in self.mental_properties 
            for p in self.material_properties
        )
        if not has_shared_properties:
            self.interaction_problems.append("No shared properties between mind and matter")
            return False
        return True

class HobbesianMaterialism:
    """Models Hobbes' materialistic account of mind and cognition"""
    
    def __init__(self):
        self.computational_properties = self._initialize_computational_properties()
        
    def _initialize_computational_properties(self) -> List[Property]:
        """Initialize properties relevant to computational materialism"""
        return [
            Property("computation", PropertyType.PRIMARY, True, False, "Information processing"),
            Property("state_transitions", PropertyType.PRIMARY, True, False, "Mechanical changes"),
            Property("symbol_manipulation", PropertyType.PRIMARY, True, False, "Formal operations")
        ]
    
    def model_cognition(self, material_state: MaterialState) -> Optional[MentalState]:
        """Attempt to derive mental properties from material properties"""
        try:
            # Hobbes' attempt to derive mind from matter through computation
            computational_state = self._compute_mental_from_material(material_state)
            return computational_state
        except Exception as e:
            # Represents Descartes' objection that material computation cannot yield mind
            return None
            
    def _compute_mental_from_material(self, material_state: MaterialState) -> MentalState:
        """Attempt to compute mental properties from material properties"""
        # Hobbes' computational theory of mind
        return MentalState(
            consciousness=0.0,  # Cannot derive consciousness
            rationality=[],  # Cannot derive genuine rationality
            qualia=[],  # Cannot derive qualia
            certainty=0.0  # Cannot derive Cartesian certainty
        )

class CartesianHobbesianDebate:
    """Models the debate between Cartesian dualism and Hobbesian materialism"""
    
    def __init__(self):
        self.cartesian_model = CartesianMindMatterDivide()
        self.hobbesian_model = HobbesianMaterialism()
        self.key_issues = self._initialize_key_issues()
        
    def _initialize_key_issues(self) -> List[str]:
        """Initialize the key points of contention in the debate"""
        return [
            "Nature of consciousness",
            "Possibility of artificial intelligence",
            "Source of meaning and purpose",
            "Basis of rationality",
            "Mind-body interaction",
            "Nature of qualia",
            "Source of certainty"
        ]
    
    def evaluate_position(self, position: str, mental_state: MentalState, 
                         material_state: MaterialState) -> Dict[str, bool]:
        """Evaluate how well each position handles key issues"""
        results = {}
        
        if position == "cartesian":
            # Evaluate Cartesian solutions
            results["consciousness"] = mental_state.consciousness > 0
            results["meaning"] = len(mental_state.rationality) > 0
            results["interaction"] = self.cartesian_model.assess_interaction_possibility(
                mental_state, material_state
            )
            
        elif position == "hobbesian":
            # Evaluate Hobbesian solutions
            computed_mind = self.hobbesian_model.model_cognition(material_state)
            results["consciousness"] = computed_mind is not None
            results["meaning"] = computed_mind is not None
            results["interaction"] = True  # Hobbesian materialism has no interaction problem
            
        return results 