from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class ConstraintType(Enum):
    ENABLING = "enabling"  # Generates new possibilities
    SELECTIVE = "selective"  # Reduces available options

class VirtueDomain(Enum):
    COURAGE = "courage"  # Balance between cowardice and foolhardiness
    TEMPERANCE = "temperance"  # Balance between indulgence and insensitivity
    JUSTICE = "justice"  # Balance between selfishness and self-sacrifice
    WISDOM = "wisdom"  # Balance between ignorance and overthinking

@dataclass
class Constraint:
    """Represents a condition that shapes possibilities"""
    type: ConstraintType
    domain: str
    effects: List[str]
    conditions: List[str]

@dataclass
class VirtualEngine:
    """Represents a system of constraints regulating development"""
    enabling_constraints: List[Constraint]
    selective_constraints: List[Constraint]
    feedback_cycles: List[str]
    current_state: str
    target_state: str

@dataclass
class Character:
    """Represents the cultivated aspects of a person"""
    virtues: List[VirtueDomain]
    virtual_engines: List[VirtualEngine]
    development_level: float
    potential_realization: float

class AristotelianDevelopmentManager:
    """Manages development according to Aristotle's framework"""
    
    def __init__(self):
        self.virtue_template = {
            VirtueDomain.COURAGE: {
                "deficiency": "cowardice",
                "excess": "foolhardiness",
                "mean": "courage"
            },
            VirtueDomain.TEMPERANCE: {
                "deficiency": "insensitivity",
                "excess": "indulgence",
                "mean": "temperance"
            },
            VirtueDomain.JUSTICE: {
                "deficiency": "selfishness",
                "excess": "self-sacrifice",
                "mean": "justice"
            },
            VirtueDomain.WISDOM: {
                "deficiency": "ignorance",
                "excess": "overthinking",
                "mean": "wisdom"
            }
        }
    
    def create_virtual_engine(self,
                            domain: str,
                            current: str,
                            target: str) -> VirtualEngine:
        """Creates a virtual engine for development in a domain"""
        return VirtualEngine(
            enabling_constraints=[],
            selective_constraints=[],
            feedback_cycles=[],
            current_state=current,
            target_state=target
        )
    
    def evaluate_character(self,
                         virtues: List[VirtueDomain],
                         engines: List[VirtualEngine]) -> Character:
        """Evaluates character based on virtues and engines"""
        # Calculate development level based on virtues
        development = len(virtues) / len(VirtueDomain)
        
        # Calculate potential realization
        potential = sum(1 for engine in engines 
                       if engine.current_state == engine.target_state)
        potential /= len(engines) if engines else 1
        
        return Character(
            virtues=virtues,
            virtual_engines=engines,
            development_level=development,
            potential_realization=potential
        )
    
    def find_golden_mean(self,
                        virtue: VirtueDomain,
                        current_state: float) -> float:
        """Finds the golden mean for a virtue"""
        template = self.virtue_template[virtue]
        
        # 0 represents complete deficiency
        # 0.5 represents the mean
        # 1 represents complete excess
        if current_state < 0.5:
            # Need to move toward mean from deficiency
            adjustment = (0.5 - current_state) * 0.1
        else:
            # Need to move toward mean from excess
            adjustment = (current_state - 0.5) * -0.1
            
        return current_state + adjustment
    
    def update_constraints(self,
                         engine: VirtualEngine,
                         new_conditions: List[str]) -> VirtualEngine:
        """Updates constraints based on new conditions"""
        # Add enabling constraints for positive conditions
        engine.enabling_constraints.extend([
            Constraint(
                type=ConstraintType.ENABLING,
                domain=engine.current_state,
                effects=["increased_possibility"],
                conditions=[cond]
            )
            for cond in new_conditions if "positive" in cond
        ])
        
        # Add selective constraints for limiting conditions
        engine.selective_constraints.extend([
            Constraint(
                type=ConstraintType.SELECTIVE,
                domain=engine.current_state,
                effects=["reduced_options"],
                conditions=[cond]
            )
            for cond in new_conditions if "limiting" in cond
        ])
        
        return engine 