from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional

class CollapseDriver(Enum):
    BLACK_DEATH = auto()    # Plague and societal disruption
    COMMERCIALISM = auto()  # Rise of trade and banking
    NOMINALISM = auto()     # Shift in understanding universals
    INQUISITION = auto()    # Institutional control and heresy hunting

class SocialDomain(Enum):
    CHURCH = auto()      # Religious institutions
    ARISTOCRACY = auto() # Traditional power structures
    MERCHANTS = auto()   # Rising commercial class
    PEASANTS = auto()    # Agricultural workers

@dataclass
class InstitutionalState:
    """Models the state of medieval institutions."""
    authority: float  # Level of institutional authority
    viability: float  # Institutional viability
    adaptation: float  # Capacity to adapt to changes

@dataclass
class WorldviewComponent:
    """Represents components of the medieval worldview."""
    coherence: float  # Internal consistency
    social_support: float  # Support from social structures
    experiential_viability: float  # Lived experience viability

class MedievalCollapse:
    """Models the collapse of the medieval worldview."""
    
    def __init__(self):
        self.institutions = {
            SocialDomain.CHURCH: InstitutionalState(1.0, 1.0, 0.5),
            SocialDomain.ARISTOCRACY: InstitutionalState(0.9, 0.8, 0.3),
            SocialDomain.MERCHANTS: InstitutionalState(0.3, 0.6, 0.9),
            SocialDomain.PEASANTS: InstitutionalState(0.2, 0.4, 0.6)
        }
        self.collapse_impacts = {
            CollapseDriver.BLACK_DEATH: 0.0,
            CollapseDriver.COMMERCIALISM: 0.0,
            CollapseDriver.NOMINALISM: 0.0,
            CollapseDriver.INQUISITION: 0.0
        }
        self.transformations: List[str] = []

    def apply_black_death_impact(self) -> None:
        """Models the impact of the Black Death."""
        self.collapse_impacts[CollapseDriver.BLACK_DEATH] = 0.9
        for domain in SocialDomain:
            self.institutions[domain].viability *= 0.5
            self.institutions[domain].authority *= 0.7
        self.transformations.append(
            "Black Death disrupts social order and institutional authority"
        )

    def apply_commercial_revolution(self) -> None:
        """Models the rise of commercialism and new institutions."""
        self.collapse_impacts[CollapseDriver.COMMERCIALISM] = 0.8
        self.institutions[SocialDomain.MERCHANTS].authority += 0.4
        self.institutions[SocialDomain.MERCHANTS].viability += 0.3
        self.institutions[SocialDomain.ARISTOCRACY].authority *= 0.8
        self.transformations.append(
            "Commercial revolution empowers merchant class"
        )

@dataclass
class SecularDynamics:
    """Models the emergence of secular institutions and practices."""
    commercial_power: float  # Power of commercial institutions
    state_autonomy: float   # Independence from religious authority
    rational_authority: float  # Authority of rational/scientific thinking

class PsychoTechnology:
    """Models the transformation of psycho-technologies."""
    
    def __init__(self):
        self.reading_mode: str = "participatory"  # participatory/silent/analytical
        self.mathematical_capability: float = 0.1  # Sophistication of mathematical tools
        self.observational_precision: float = 0.2  # Precision of natural observation
        
    def advance_mathematical_tools(self) -> None:
        """Models the adoption of new mathematical tools."""
        self.mathematical_capability += 0.5
        self.reading_mode = "analytical"

class MedievalTransformationTracker:
    """Tracks the overall transformation of medieval society and worldview."""
    
    def __init__(self):
        self.collapse = MedievalCollapse()
        self.secular = SecularDynamics(0.1, 0.1, 0.1)
        self.psychotech = PsychoTechnology()
        self.historical_stages: List[str] = []

    def advance_transformation(self) -> None:
        """Advances the transformation through its key stages."""
        # Apply Black Death impact
        self.collapse.apply_black_death_impact()
        self.secular.state_autonomy += 0.2
        
        # Apply Commercial Revolution
        self.collapse.apply_commercial_revolution()
        self.secular.commercial_power += 0.4
        self.secular.state_autonomy += 0.3
        
        # Advance psycho-technologies
        self.psychotech.advance_mathematical_tools()
        self.secular.rational_authority += 0.4
        
        self.historical_stages.append(
            "Medieval worldview transformation through crisis and innovation"
        )

    def assess_transformation_state(self) -> Dict[str, float]:
        """Assesses the current state of medieval transformation."""
        return {
            "institutional_viability": 
                sum(i.viability for i in self.collapse.institutions.values()) / 4,
            "secular_emergence":
                (self.secular.commercial_power + 
                 self.secular.state_autonomy + 
                 self.secular.rational_authority) / 3,
            "psychotech_advancement":
                (self.psychotech.mathematical_capability + 
                 self.psychotech.observational_precision) / 2,
            "overall_transformation":
                sum(self.collapse.collapse_impacts.values()) / 4
        } 