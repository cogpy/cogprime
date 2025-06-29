from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import math

class DiscountingType(Enum):
    HYPERBOLIC = "hyperbolic"  # Natural discounting curve
    EXPONENTIAL = "exponential"  # Rational discounting curve
    MIXED = "mixed"  # Combination of both

class TimeHorizon(Enum):
    IMMEDIATE = "immediate"  # Present moment
    SHORT_TERM = "short_term"  # Days to weeks
    MEDIUM_TERM = "medium_term"  # Months to years
    LONG_TERM = "long_term"  # Years to decades

class AbstractionLevel(Enum):
    CONCRETE = "concrete"  # Specific instances
    CATEGORICAL = "categorical"  # Classes of things
    ABSTRACT = "abstract"  # General principles
    UNIVERSAL = "universal"  # Universal patterns

@dataclass
class DiscountingParameters:
    """Parameters for discounting calculations"""
    k: float  # Discounting rate
    time_scale: float  # Time scaling factor
    baseline_salience: float  # Initial salience level
    abstraction_penalty: float  # Penalty for abstract concepts

@dataclass
class ProbabilityChain:
    """Represents a chain of probabilistic events"""
    events: List[str]
    probabilities: List[float]
    cumulative_probability: float
    abstract_pattern: str

class HyperbolicDiscountingManager:
    """Manages hyperbolic discounting calculations and analysis"""
    
    def __init__(self):
        self.default_params = DiscountingParameters(
            k=0.1,
            time_scale=1.0,
            baseline_salience=1.0,
            abstraction_penalty=0.2
        )
        
        # Example of smoking consequences
        self.smoking_chain = ProbabilityChain(
            events=[
                "immediate_pleasure",
                "short_term_cough",
                "medium_term_health_decline",
                "long_term_cancer"
            ],
            probabilities=[1.0, 0.5, 0.25, 0.125],
            cumulative_probability=0.015625,  # Product of probabilities
            abstract_pattern="premature_death"
        )
    
    def calculate_discounted_value(self,
                                 value: float,
                                 delay: float,
                                 params: Optional[DiscountingParameters] = None) -> float:
        """Calculates discounted value using hyperbolic function"""
        if params is None:
            params = self.default_params
            
        return value / (1 + params.k * delay * params.time_scale)
    
    def calculate_salience(self,
                         base_value: float,
                         abstraction_level: AbstractionLevel,
                         time_horizon: TimeHorizon,
                         params: Optional[DiscountingParameters] = None) -> float:
        """Calculates salience considering both time and abstraction"""
        if params is None:
            params = self.default_params
            
        # Time penalty
        time_penalties = {
            TimeHorizon.IMMEDIATE: 0.0,
            TimeHorizon.SHORT_TERM: 0.2,
            TimeHorizon.MEDIUM_TERM: 0.5,
            TimeHorizon.LONG_TERM: 0.8
        }
        time_penalty = time_penalties[time_horizon]
        
        # Abstraction penalty
        abstraction_penalties = {
            AbstractionLevel.CONCRETE: 0.0,
            AbstractionLevel.CATEGORICAL: 0.3,
            AbstractionLevel.ABSTRACT: 0.6,
            AbstractionLevel.UNIVERSAL: 0.9
        }
        abstraction_penalty = abstraction_penalties[abstraction_level]
        
        # Combined discount
        total_penalty = (time_penalty + abstraction_penalty * params.abstraction_penalty)
        return base_value * math.exp(-total_penalty)
    
    def analyze_probability_chain(self,
                                chain: ProbabilityChain,
                                time_scale: float = 1.0) -> dict:
        """Analyzes a chain of probabilistic events"""
        results = {
            "individual_events": [],
            "cumulative_probability": chain.cumulative_probability,
            "abstract_pattern": chain.abstract_pattern,
            "salience_profile": []
        }
        
        for i, (event, prob) in enumerate(zip(chain.events, chain.probabilities)):
            discounted_prob = self.calculate_discounted_value(
                prob, i * time_scale
            )
            results["individual_events"].append({
                "event": event,
                "raw_probability": prob,
                "discounted_probability": discounted_prob
            })
            
            # Calculate salience for each event
            if i == 0:
                horizon = TimeHorizon.IMMEDIATE
            elif i == 1:
                horizon = TimeHorizon.SHORT_TERM
            elif i == 2:
                horizon = TimeHorizon.MEDIUM_TERM
            else:
                horizon = TimeHorizon.LONG_TERM
                
            salience = self.calculate_salience(
                prob,
                AbstractionLevel.CONCRETE,
                horizon
            )
            results["salience_profile"].append({
                "event": event,
                "salience": salience
            })
            
        return results
    
    def evaluate_self_deception(self,
                              concrete_salience: float,
                              abstract_importance: float) -> float:
        """Evaluates potential for self-deception based on salience-importance gap"""
        return abs(concrete_salience - abstract_importance)
    
    def get_smoking_example(self) -> ProbabilityChain:
        """Returns the example probability chain for smoking"""
        return self.smoking_chain 