from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class PsychePart(Enum):
    REASON = "reason"  # The man in the head - truth and abstract thought
    THYMOS = "thymos"  # The lion in the chest - honor and social motivation
    APPETITE = "appetite"  # The monster in the belly - pleasure and immediate needs

class CognitiveDomain(Enum):
    ABSTRACT = "abstract"  # Long-term, theoretical understanding
    SOCIAL = "social"  # Mid-term, cultural meaning
    IMMEDIATE = "immediate"  # Short-term, sensory pleasure

class TransformationStage(Enum):
    CHAINED = "chained"  # Initial state of illusion
    FREED = "freed"  # First step of liberation
    FIRE_SIGHT = "fire_sight"  # Seeing the source of shadows
    ASCENT = "ascent"  # Journey upward
    BLINDING = "blinding"  # Temporary disorientation
    ADJUSTMENT = "adjustment"  # New level of seeing
    ENLIGHTENMENT = "enlightenment"  # Final realization

@dataclass
class PsychicState:
    """Represents the state of the psyche"""
    reason_strength: float
    thymos_alignment: float
    appetite_control: float
    inner_conflict: float
    reality_contact: float

@dataclass
class TransformationProcess:
    """Represents the process of transformation in the cave allegory"""
    current_stage: TransformationStage
    insight_level: float
    self_transcendence: float
    reality_grasp: float
    next_challenge: str

class PlatonicPsycheManager:
    """Manages the Platonic model of the psyche and transformation"""
    
    def __init__(self):
        self.psyche_template = PsychicState(
            reason_strength=0.0,
            thymos_alignment=0.0,
            appetite_control=0.0,
            inner_conflict=1.0,
            reality_contact=0.0
        )
        
        self.transformation_template = TransformationProcess(
            current_stage=TransformationStage.CHAINED,
            insight_level=0.0,
            self_transcendence=0.0,
            reality_grasp=0.0,
            next_challenge=""
        )
    
    def evaluate_psyche(self, 
                       reason: float,
                       thymos: float,
                       appetite: float) -> PsychicState:
        """Evaluates the state of the psyche based on its parts"""
        state = self.psyche_template
        state.reason_strength = reason
        state.thymos_alignment = thymos
        state.appetite_control = appetite
        
        # Calculate inner conflict as inverse of harmony
        harmony = min(reason, thymos, appetite)
        state.inner_conflict = 1.0 - harmony
        
        # Calculate reality contact based on reduced self-deception
        state.reality_contact = (reason * thymos * appetite) ** (1/3)
        
        return state
    
    def advance_transformation(self, 
                             current: TransformationProcess,
                             insight_gain: float) -> TransformationProcess:
        """Advances the transformation process based on new insights"""
        process = current
        
        # Update metrics
        process.insight_level += insight_gain
        process.self_transcendence = process.insight_level * 0.8
        process.reality_grasp = process.insight_level * 0.7
        
        # Determine next stage
        if process.insight_level >= 0.8:
            process.current_stage = TransformationStage.ENLIGHTENMENT
            process.next_challenge = "maintain_vision"
        elif process.insight_level >= 0.6:
            process.current_stage = TransformationStage.ADJUSTMENT
            process.next_challenge = "integrate_insight"
        elif process.insight_level >= 0.4:
            process.current_stage = TransformationStage.BLINDING
            process.next_challenge = "endure_transformation"
        elif process.insight_level >= 0.2:
            process.current_stage = TransformationStage.ASCENT
            process.next_challenge = "continue_journey"
        else:
            process.current_stage = TransformationStage.FIRE_SIGHT
            process.next_challenge = "question_shadows"
            
        return process
    
    def calculate_wisdom(self, psyche: PsychicState) -> float:
        """Calculates wisdom level based on psyche state"""
        return (
            (1 - psyche.inner_conflict) * 0.5 +
            psyche.reality_contact * 0.5
        )
    
    def get_cognitive_domain(self, part: PsychePart) -> CognitiveDomain:
        """Returns the cognitive domain for a psyche part"""
        domain_map = {
            PsychePart.REASON: CognitiveDomain.ABSTRACT,
            PsychePart.THYMOS: CognitiveDomain.SOCIAL,
            PsychePart.APPETITE: CognitiveDomain.IMMEDIATE
        }
        return domain_map[part] 