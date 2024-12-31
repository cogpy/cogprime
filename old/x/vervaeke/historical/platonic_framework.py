from dataclasses import dataclass
from typing import List, Optional, Dict
from .platonic_psyche import PlatonicPsycheManager, PsychicState, TransformationProcess
from .platonic_forms import PlatonicFormManager, Form, KnowingMode
from .hyperbolic_discounting import HyperbolicDiscountingManager, ProbabilityChain

@dataclass
class WisdomState:
    """Represents the overall state of wisdom development"""
    psychic_state: PsychicState
    transformation_process: TransformationProcess
    active_forms: List[Form]
    knowing_modes: List[KnowingMode]
    self_deception_level: float

class PlatonicFramework:
    """High-level interface for Plato's psychological and philosophical framework"""
    
    def __init__(self):
        self.psyche_manager = PlatonicPsycheManager()
        self.form_manager = PlatonicFormManager()
        self.discounting_manager = HyperbolicDiscountingManager()
        
        self.wisdom_state = WisdomState(
            psychic_state=self.psyche_manager.psyche_template,
            transformation_process=self.psyche_manager.transformation_template,
            active_forms=[],
            knowing_modes=[],
            self_deception_level=1.0
        )
    
    def evaluate_current_state(self) -> WisdomState:
        """Evaluates current state of wisdom development"""
        return self.wisdom_state
    
    def advance_development(self,
                          reason_development: float,
                          thymos_training: float,
                          appetite_taming: float,
                          insight_gain: float) -> WisdomState:
        """Advances overall development based on progress in different areas"""
        # Update psychic state
        new_psyche = self.psyche_manager.evaluate_psyche(
            reason_development,
            thymos_training,
            appetite_taming
        )
        
        # Update transformation process
        new_transformation = self.psyche_manager.advance_transformation(
            self.wisdom_state.transformation_process,
            insight_gain
        )
        
        # Update forms and knowing modes based on development
        new_forms = []
        new_modes = []
        if reason_development > 0.6:
            bird_form = self.form_manager.get_bird_form()
            new_forms.append(bird_form)
            new_modes.append(KnowingMode.LOGOS_UNDERSTANDING)
        
        # Calculate new self-deception level
        concrete_salience = self.discounting_manager.calculate_salience(
            1.0,  # base value
            self.discounting_manager.AbstractionLevel.CONCRETE,
            self.discounting_manager.TimeHorizon.IMMEDIATE
        )
        abstract_importance = reason_development
        new_self_deception = self.discounting_manager.evaluate_self_deception(
            concrete_salience,
            abstract_importance
        )
        
        # Update wisdom state
        self.wisdom_state = WisdomState(
            psychic_state=new_psyche,
            transformation_process=new_transformation,
            active_forms=new_forms,
            knowing_modes=new_modes,
            self_deception_level=new_self_deception
        )
        
        return self.wisdom_state
    
    def analyze_decision_process(self,
                               immediate_value: float,
                               abstract_value: float,
                               time_horizon: str) -> Dict:
        """Analyzes a decision process using the framework"""
        # Create probability chain
        chain = ProbabilityChain(
            events=["immediate_choice", "future_consequence"],
            probabilities=[immediate_value, abstract_value],
            cumulative_probability=immediate_value * abstract_value,
            abstract_pattern="value_realization"
        )
        
        # Analyze using hyperbolic discounting
        discounting_analysis = self.discounting_manager.analyze_probability_chain(
            chain
        )
        
        # Get current psychic state
        psyche_state = self.psyche_manager.evaluate_psyche(
            self.wisdom_state.psychic_state.reason_strength,
            self.wisdom_state.psychic_state.thymos_alignment,
            self.wisdom_state.psychic_state.appetite_control
        )
        
        return {
            "discounting_analysis": discounting_analysis,
            "psyche_state": psyche_state,
            "wisdom_level": self.psyche_manager.calculate_wisdom(psyche_state)
        }
    
    def get_development_recommendations(self) -> List[str]:
        """Provides recommendations for further development"""
        recommendations = []
        
        # Check psychic state
        if self.wisdom_state.psychic_state.reason_strength < 0.4:
            recommendations.append("Strengthen rational understanding")
        if self.wisdom_state.psychic_state.thymos_alignment < 0.4:
            recommendations.append("Develop social awareness")
        if self.wisdom_state.psychic_state.appetite_control < 0.4:
            recommendations.append("Practice appetite regulation")
            
        # Check transformation stage
        if "endure_transformation" in self.wisdom_state.transformation_process.next_challenge:
            recommendations.append("Focus on enduring disorientation")
        elif "integrate_insight" in self.wisdom_state.transformation_process.next_challenge:
            recommendations.append("Work on integrating new insights")
            
        # Check self-deception
        if self.wisdom_state.self_deception_level > 0.6:
            recommendations.append("Address gap between immediate and abstract understanding")
            
        return recommendations 