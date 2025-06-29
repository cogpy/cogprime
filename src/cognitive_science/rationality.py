"""
Module for handling rationality components based on Vervaeke's framework.
Implements mechanisms for systematic overcoming of self-deception.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

class ErrorType(Enum):
    COMPETENCE = "competence"
    PERFORMANCE = "performance"
    IMPLEMENTATION = "implementation"

class RationalityDomain(Enum):
    DAILY_LIFE = "daily_life"
    PROFESSIONAL = "professional"
    FRIENDSHIP = "friendship"
    ROMANTIC = "romantic"
    MORAL = "moral"

@dataclass
class RationalityError:
    """Represents a failure of rational processing."""
    type: ErrorType
    domain: RationalityDomain
    description: str
    impact: Dict[str, float]
    frequency: float
    systematic: bool

@dataclass
class RationalityPrinciple:
    """Represents a principle of rational processing."""
    description: str
    domains: Set[RationalityDomain]
    acknowledged: bool
    applied: bool
    success_rate: float

class RationalityCore:
    """Core class for managing rationality development."""
    
    def __init__(self):
        self.errors: List[RationalityError] = []
        self.principles: Dict[str, RationalityPrinciple] = {}
        self.domain_performance: Dict[RationalityDomain, float] = {
            domain: 0.0 for domain in RationalityDomain
        }
        
    def register_error(self, error: RationalityError) -> None:
        """Register a rational processing error."""
        self.errors.append(error)
        self._update_domain_performance(error.domain, -error.impact.get("severity", 0.0))
        
    def add_principle(self, principle: RationalityPrinciple) -> None:
        """Add a rationality principle."""
        self.principles[principle.description] = principle
        
    def _update_domain_performance(self, domain: RationalityDomain, impact: float) -> None:
        """Update performance in a rationality domain."""
        current = self.domain_performance[domain]
        self.domain_performance[domain] = max(0.0, min(1.0, current + impact))
        
    def analyze_competence_performance_gap(self) -> Dict[str, float]:
        """Analyze the gap between competence and performance."""
        gaps = {}
        for principle in self.principles.values():
            if principle.acknowledged and not principle.applied:
                for domain in principle.domains:
                    key = f"{domain.value}:{principle.description}"
                    gaps[key] = 1.0 - principle.success_rate
        return gaps
        
    def evaluate_systematic_rationality(self) -> Dict[str, float]:
        """Evaluate rationality across different domains."""
        systematic_score = sum(
            1.0 for p in self.principles.values()
            if p.acknowledged and p.applied and p.success_rate > 0.7
        ) / max(len(self.principles), 1)
        
        domain_scores = {
            domain.value: score 
            for domain, score in self.domain_performance.items()
        }
        
        error_rate = sum(
            1.0 for e in self.errors 
            if e.type == ErrorType.PERFORMANCE
        ) / max(len(self.errors), 1)
        
        return {
            "systematic_rationality": systematic_score,
            "domain_coverage": len([s for s in domain_scores.values() if s > 0.5]) / len(RationalityDomain),
            "performance_error_rate": error_rate,
            **domain_scores
        }
        
    def optimize_processing(self) -> List[str]:
        """Generate recommendations for optimizing rational processing."""
        recommendations = []
        
        # Check for systematic errors
        systematic_errors = [e for e in self.errors if e.systematic]
        if systematic_errors:
            recommendations.append(
                "Address systematic errors in domains: " + 
                ", ".join(set(e.domain.value for e in systematic_errors))
            )
            
        # Check for principle application gaps
        principle_gaps = [
            p for p in self.principles.values()
            if p.acknowledged and not p.applied
        ]
        if principle_gaps:
            recommendations.append(
                "Focus on applying acknowledged principles: " +
                ", ".join(p.description for p in principle_gaps)
            )
            
        # Check for domain weaknesses
        weak_domains = [
            domain for domain, score in self.domain_performance.items()
            if score < 0.3
        ]
        if weak_domains:
            recommendations.append(
                "Strengthen rationality in domains: " +
                ", ".join(d.value for d in weak_domains)
            )
            
        return recommendations 