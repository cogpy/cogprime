from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

class EvidenceType(Enum):
    SENSORY = "sensory"           # Direct sensory evidence
    EXPERIMENTAL = "experimental"  # Scientific experimental evidence
    LOGICAL = "logical"           # Logical/mathematical proof
    CONVERGENT = "convergent"     # Multiple independent lines
    PRACTICAL = "practical"       # Practical effectiveness

class PlausibilityAspect(Enum):
    TRUSTWORTHINESS = "trustworthiness"  # Reduced self-deception
    ELEGANCE = "elegance"                # Multi-apt applicability
    FLUENCY = "fluency"                  # Processing efficiency
    CONVERGENCE = "convergence"          # Independent support
    POWER = "power"                      # Explanatory/predictive power

@dataclass
class EvidenceLine:
    """Represents a line of evidence supporting a claim"""
    type: EvidenceType
    strength: float  # 0-1 scale
    independence: float  # How independent from other lines
    reliability: float  # How reliable/repeatable
    scope: List[str]  # What domains it applies to

@dataclass
class PlausibilityAssessment:
    """Represents the plausibility assessment of a claim/model"""
    trustworthiness: float
    elegance: float
    fluency: float
    convergence: float
    power: float
    overall_plausibility: float

class PlausibilityManager:
    """Manages assessment of plausibility in cognitive processing"""
    
    def __init__(self):
        self.evidence_lines: List[EvidenceLine] = []
        self.current_assessment = PlausibilityAssessment(
            trustworthiness=0.0,
            elegance=0.0,
            fluency=0.0,
            convergence=0.0,
            power=0.0,
            overall_plausibility=0.0
        )
        self.assessment_history: List[PlausibilityAssessment] = []

    def add_evidence(self, 
                    type: EvidenceType,
                    strength: float,
                    independence: float,
                    reliability: float,
                    scope: List[str]) -> None:
        """Adds a new line of evidence"""
        evidence = EvidenceLine(
            type=type,
            strength=strength,
            independence=independence,
            reliability=reliability,
            scope=scope
        )
        self.evidence_lines.append(evidence)
        self._update_assessment()

    def _calculate_trustworthiness(self) -> float:
        """Calculates trustworthiness based on evidence convergence"""
        if not self.evidence_lines:
            return 0.0
            
        # Weight by independence and reliability
        weighted_sum = sum(
            e.strength * e.independence * e.reliability 
            for e in self.evidence_lines
        )
        return weighted_sum / len(self.evidence_lines)

    def _calculate_elegance(self) -> float:
        """Calculates elegance based on scope and applicability"""
        if not self.evidence_lines:
            return 0.0
            
        # Count unique domains of application
        all_domains = set()
        for evidence in self.evidence_lines:
            all_domains.update(evidence.scope)
            
        return len(all_domains) / 10.0  # Normalize to 0-1 scale

    def _calculate_convergence(self) -> float:
        """Calculates convergence of independent lines"""
        if not self.evidence_lines:
            return 0.0
            
        # Average independence weighted by strength
        return sum(
            e.independence * e.strength for e in self.evidence_lines
        ) / len(self.evidence_lines)

    def _calculate_power(self) -> float:
        """Calculates explanatory/predictive power"""
        if not self.evidence_lines:
            return 0.0
            
        # Consider strength and scope breadth
        return sum(
            e.strength * len(e.scope) for e in self.evidence_lines
        ) / (len(self.evidence_lines) * 5.0)  # Normalize

    def _update_assessment(self) -> None:
        """Updates the overall plausibility assessment"""
        self.current_assessment.trustworthiness = self._calculate_trustworthiness()
        self.current_assessment.elegance = self._calculate_elegance()
        self.current_assessment.convergence = self._calculate_convergence()
        self.current_assessment.power = self._calculate_power()
        
        # Overall plausibility balances convergence and power
        balance = min(self.current_assessment.convergence, 
                     self.current_assessment.power)
        self.current_assessment.overall_plausibility = (
            balance * self.current_assessment.trustworthiness *
            self.current_assessment.elegance
        )
        
        # Record assessment
        self.assessment_history.append(PlausibilityAssessment(
            **vars(self.current_assessment)
        ))

    def evaluate_plausibility(self) -> Dict[str, float]:
        """Evaluates the current plausibility assessment"""
        return {
            "trustworthiness": self.current_assessment.trustworthiness,
            "elegance": self.current_assessment.elegance,
            "fluency": self.current_assessment.fluency,
            "convergence": self.current_assessment.convergence,
            "power": self.current_assessment.power,
            "overall_plausibility": self.current_assessment.overall_plausibility
        }

    def get_improvement_suggestions(self) -> List[str]:
        """Provides suggestions for improving plausibility"""
        suggestions = []
        
        if self.current_assessment.trustworthiness < 0.4:
            suggestions.append("Seek more independent evidence sources")
        if self.current_assessment.elegance < 0.4:
            suggestions.append("Explore applications in new domains")
        if self.current_assessment.convergence < 0.4:
            suggestions.append("Validate through different methods")
        if self.current_assessment.power < 0.4:
            suggestions.append("Strengthen explanatory framework")
            
        # Check evidence type coverage
        covered_types = {e.type for e in self.evidence_lines}
        missing_types = set(EvidenceType) - covered_types
        if missing_types:
            suggestions.append(
                f"Consider gathering {', '.join(t.value)} evidence"
            )
            
        return suggestions 