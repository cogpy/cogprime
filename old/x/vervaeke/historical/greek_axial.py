from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class GreekInnovation(Enum):
    VOWELS = "vowels"  # Enhanced cognitive fluency through vowels
    LEFT_TO_RIGHT = "left_to_right"  # Standardized reading direction
    MATHEMATICS = "mathematics"  # Abstract symbol systems
    RATIONAL_DEBATE = "rational_debate"  # Democratic argumentation
    GEOMETRY = "geometry"  # Systematic spatial reasoning

class ShamanicElement(Enum):
    SOUL_FLIGHT = "soul_flight"  # Experience of psyche liberation
    CAVE_ISOLATION = "cave_isolation"  # Transformative isolation
    DIVINE_APPEARANCE = "divine_appearance"  # God-like presence
    MUSICAL_INSIGHT = "musical_insight"  # Mathematical harmony

@dataclass
class PythagoreanSynthesis:
    """Represents Pythagoras's integration of shamanic and rational elements"""
    shamanic_elements: List[ShamanicElement]
    mathematical_insights: List[str]
    transformative_practices: List[str]
    key_concepts: List[str]

@dataclass
class SocraticMethod:
    """Represents the core elements of Socratic dialectic"""
    questioning_types: List[str]
    dialogue_structure: List[str]
    key_insights: List[str]
    transformation_goals: List[str]

class GreekAxialManager:
    """Manages the Greek contributions to the Axial revolution"""
    
    def __init__(self):
        self.innovations = list(GreekInnovation)
        
        self.pythagorean = PythagoreanSynthesis(
            shamanic_elements=[
                ShamanicElement.SOUL_FLIGHT,
                ShamanicElement.CAVE_ISOLATION,
                ShamanicElement.DIVINE_APPEARANCE,
                ShamanicElement.MUSICAL_INSIGHT
            ],
            mathematical_insights=[
                "octave_ratios",
                "geometric_proofs",
                "number_mysticism",
                "cosmic_harmony"
            ],
            transformative_practices=[
                "mathematical_contemplation",
                "musical_training",
                "communal_living",
                "dietary_restrictions"
            ],
            key_concepts=[
                "cosmos_as_ordered_beauty",
                "mathematical_reality",
                "soul_liberation",
                "rational_mysticism"
            ]
        )
        
        self.socratic = SocraticMethod(
            questioning_types=[
                "definitional",
                "assumptional",
                "implicational",
                "counter_example"
            ],
            dialogue_structure=[
                "initial_claim",
                "examination",
                "contradiction",
                "aporia"
            ],
            key_insights=[
                "examined_life",
                "knowledge_of_ignorance",
                "virtue_as_knowledge",
                "care_of_soul"
            ],
            transformation_goals=[
                "self_knowledge",
                "ethical_wisdom",
                "intellectual_humility",
                "rational_agency"
            ]
        )
    
    def get_cognitive_enhancements(self) -> List[str]:
        """Returns the cognitive enhancements from Greek innovations"""
        return [inn.value for inn in self.innovations]
    
    def get_pythagorean_elements(self) -> dict:
        """Returns the key elements of Pythagorean synthesis"""
        return {
            "shamanic": [el.value for el in self.pythagorean.shamanic_elements],
            "mathematical": self.pythagorean.mathematical_insights,
            "practices": self.pythagorean.transformative_practices,
            "concepts": self.pythagorean.key_concepts
        }
    
    def get_socratic_elements(self) -> dict:
        """Returns the key elements of the Socratic method"""
        return {
            "questioning": self.socratic.questioning_types,
            "dialogue": self.socratic.dialogue_structure,
            "insights": self.socratic.key_insights,
            "goals": self.socratic.transformation_goals
        } 