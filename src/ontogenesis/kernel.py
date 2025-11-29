"""
Kernel data structures for ontogenesis.

This module defines the core kernel classes including the base GeneratedKernel
and the enhanced OntogeneticKernel with genetic capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np


class DevelopmentStage(Enum):
    """Life stages of a kernel."""

    EMBRYONIC = "embryonic"  # Just generated, basic structure
    JUVENILE = "juvenile"  # Developing, optimizing
    MATURE = "mature"  # Fully developed, capable of reproduction
    SENESCENT = "senescent"  # Declining, ready for replacement


@dataclass
class GripMetrics:
    """Metrics for measuring kernel-domain fit."""

    contact: float  # How well kernel touches domain
    coverage: float  # Completeness of span
    efficiency: float  # Computational cost
    stability: float  # Numerical properties

    @property
    def total_grip(self) -> float:
        """Calculate total grip score."""
        return (
            self.contact * 0.4 + self.coverage * 0.3 + self.stability * 0.2 + self.efficiency * 0.1
        )


@dataclass
class OntogeneticState:
    """Development state of an ontogenetic kernel."""

    stage: DevelopmentStage
    maturity: float  # 0.0 to 1.0
    development_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record a development event."""
        self.development_history.append({"type": event_type, "details": details})


@dataclass
class GeneratedKernel:
    """Base kernel generated through differential calculus.

    This represents a kernel generated via B-series expansion using elementary
    differentials (rooted trees following A000081 sequence).
    """

    order: int  # Order of the differential expansion
    coefficients: np.ndarray  # B-series coefficients
    domain_spec: str  # Domain specification
    grip_metrics: GripMetrics  # Fit to domain
    trees: List[str] = field(default_factory=list)  # Elementary differentials

    def __post_init__(self):
        """Validate kernel after initialization."""
        if len(self.coefficients) == 0:
            raise ValueError("Kernel must have at least one coefficient")
        if self.order < 1:
            raise ValueError("Kernel order must be at least 1")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the kernel at point x.

        This is a simplified evaluation that computes a weighted sum of the input.
        In a full implementation, this would evaluate the B-series expansion.
        """
        # Ensure x is an array
        x = np.asarray(x)

        # Simple evaluation: weighted sum with coefficients
        if len(x.shape) == 1:
            # 1D input
            result = np.sum(self.coefficients[: len(x)] * x[: len(self.coefficients)])
        else:
            # Multi-dimensional input - use first dimension
            result = np.sum(self.coefficients[: x.shape[0]] * x[: len(self.coefficients), 0])

        return result

    def differentiate(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of kernel at point x.

        This is a simplified differentiation. In a full implementation,
        this would compute the derivative of the B-series expansion.
        """
        # Simple gradient: return coefficients as derivative
        return self.coefficients.copy()


@dataclass
class OntogeneticKernel(GeneratedKernel):
    """Enhanced kernel with genetic capabilities.

    Extends GeneratedKernel with genome and ontogenetic state for
    self-generation, self-optimization, and evolution.
    """

    genome: Optional["KernelGenome"] = None  # Genetic information (forward reference)
    ontogenetic_state: Optional[OntogeneticState] = None  # Development state

    def __post_init__(self):
        """Initialize ontogenetic kernel."""
        super().__post_init__()

        # Import here to avoid circular dependency
        from .genome import KernelGenome, CoefficientGene

        if self.genome is None:
            # Create default genome
            genes = [CoefficientGene(index=i, value=coeff) for i, coeff in enumerate(self.coefficients)]
            self.genome = KernelGenome(
                id=f"kernel_{id(self)}",
                generation=0,
                lineage=[],
                genes=genes,
                fitness=self.grip_metrics.total_grip,
                age=0,
            )

        if self.ontogenetic_state is None:
            # Create initial state
            self.ontogenetic_state = OntogeneticState(
                stage=DevelopmentStage.EMBRYONIC, maturity=0.0
            )

    def get_fitness(self) -> float:
        """Get the fitness of this kernel."""
        if self.genome:
            return self.genome.fitness
        return self.grip_metrics.total_grip

    def advance_stage(self) -> None:
        """Advance to the next development stage if mature enough."""
        if not self.ontogenetic_state:
            return

        state = self.ontogenetic_state
        stage_transitions = {
            DevelopmentStage.EMBRYONIC: (0.3, DevelopmentStage.JUVENILE),
            DevelopmentStage.JUVENILE: (0.7, DevelopmentStage.MATURE),
            DevelopmentStage.MATURE: (0.95, DevelopmentStage.SENESCENT),
        }

        if state.stage in stage_transitions:
            threshold, next_stage = stage_transitions[state.stage]
            if state.maturity >= threshold:
                old_stage = state.stage
                state.stage = next_stage
                state.record_event(
                    "stage_transition", {"from": old_stage.value, "to": next_stage.value}
                )
