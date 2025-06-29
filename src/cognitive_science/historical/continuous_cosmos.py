from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class CosmicStructure(Enum):
    CONTINUOUS = "continuous"  # Pre-axial worldview of radical continuity
    DISEMBEDDED = "disembedded"  # Post-axial worldview of two worlds

class TimeStructure(Enum):
    CYCLICAL = "cyclical"  # Time moves in repeating cycles
    NARRATIVE = "narrative"  # Time moves as a directed story with purpose

class KnowingMode(Enum):
    POWER = "power"  # Knowledge as tapping into cycles of power
    PARTICIPATORY = "participatory"  # Knowledge as participation (da'ath)
    RATIONAL = "rational"  # Greek rational argumentation

@dataclass
class WorldGrammar:
    """Represents the fundamental structure of how meaning and reality are understood"""
    cosmic_structure: CosmicStructure
    time_structure: TimeStructure
    primary_knowing: KnowingMode
    emphasis_on_progress: bool
    emphasis_on_transcendence: bool
    
@dataclass 
class AxialTransformation:
    """Represents the transformation from continuous to disembedded worldview"""
    from_grammar: WorldGrammar
    to_grammar: WorldGrammar
    psychotechnologies: List[str]
    key_insights: List[str]

class ContinuousCosmosManager:
    """Manages the transition between continuous cosmos and modern world grammar"""
    
    def __init__(self):
        self.continuous_cosmos = WorldGrammar(
            cosmic_structure=CosmicStructure.CONTINUOUS,
            time_structure=TimeStructure.CYCLICAL,
            primary_knowing=KnowingMode.POWER,
            emphasis_on_progress=False,
            emphasis_on_transcendence=False
        )
        
        self.modern_grammar = WorldGrammar(
            cosmic_structure=CosmicStructure.DISEMBEDDED,
            time_structure=TimeStructure.NARRATIVE,
            primary_knowing=KnowingMode.PARTICIPATORY,
            emphasis_on_progress=True,
            emphasis_on_transcendence=True
        )
        
        self.axial_transformation = AxialTransformation(
            from_grammar=self.continuous_cosmos,
            to_grammar=self.modern_grammar,
            psychotechnologies=[
                "alphabetic_literacy",
                "coinage",
                "rational_argumentation",
                "narrative_time"
            ],
            key_insights=[
                "self_transcendence",
                "moral_responsibility",
                "progress_possibility",
                "participatory_knowing"
            ]
        )
    
    def analyze_worldview(self, grammar: WorldGrammar) -> dict:
        """Analyzes the implications of a particular world grammar"""
        return {
            "meaning_structure": "power-based" if grammar.primary_knowing == KnowingMode.POWER else "participatory",
            "time_conception": "cyclical" if grammar.time_structure == TimeStructure.CYCLICAL else "narrative",
            "self_conception": "embedded" if grammar.cosmic_structure == CosmicStructure.CONTINUOUS else "transcendent",
            "moral_emphasis": "low" if not grammar.emphasis_on_progress else "high"
        }
    
    def get_psychotechnologies(self) -> List[str]:
        """Returns the key psychotechnologies developed in the axial transformation"""
        return self.axial_transformation.psychotechnologies
    
    def get_key_insights(self) -> List[str]:
        """Returns the key insights that emerged from the axial transformation"""
        return self.axial_transformation.key_insights 