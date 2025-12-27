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
    telos: Optional["Telos"] = None  # Intrinsic purpose (forward reference)
    actualization_tracker: Optional["ActualizationTracker"] = None  # Tracks actualization progress

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
        
        # Initialize actualization tracker if not provided
        if self.actualization_tracker is None:
            try:
                from ..telos import ActualizationTracker
                self.actualization_tracker = ActualizationTracker()
            except ImportError:
                pass  # Telos module not available

    def get_fitness(self) -> float:
        """Get the fitness of this kernel."""
        if self.genome:
            return self.genome.fitness
        return self.grip_metrics.total_grip

    def advance_stage(self) -> None:
        """Advance to the next development stage if mature enough.
        
        Uses teleological guidance if telos is available.
        """
        if not self.ontogenetic_state:
            return

        state = self.ontogenetic_state
        stage_transitions = {
            DevelopmentStage.EMBRYONIC: (0.3, DevelopmentStage.JUVENILE),
            DevelopmentStage.JUVENILE: (0.7, DevelopmentStage.MATURE),
            DevelopmentStage.MATURE: (0.95, DevelopmentStage.SENESCENT),
        }

        # Check if ready for transition
        if state.stage in stage_transitions:
            threshold, next_stage = stage_transitions[state.stage]
            
            # Adjust threshold based on telos actualization if available
            if self.telos is not None and self.actualization_tracker is not None:
                try:
                    metrics = self.actualization_tracker.compute_metrics(self, self.telos)
                    # Lower threshold if actualization is high (ready to advance)
                    # Raise threshold if actualization is low (need more development)
                    adjustment = (metrics.actualization - 0.5) * 0.2  # ±0.1 adjustment
                    adjusted_threshold = threshold - adjustment
                    adjusted_threshold = max(0.1, min(0.95, adjusted_threshold))
                except:
                    adjusted_threshold = threshold
            else:
                adjusted_threshold = threshold
            
            if state.maturity >= adjusted_threshold:
                old_stage = state.stage
                state.stage = next_stage
                state.record_event(
                    "stage_transition", {
                        "from": old_stage.value, 
                        "to": next_stage.value,
                        "maturity": state.maturity,
                        "threshold": adjusted_threshold
                    }
                )
                
                # Record in actualization tracker if available
                if self.actualization_tracker is not None:
                    transition = self.actualization_tracker.detect_phase_transition()
                    if transition:
                        state.record_event("phase_transition", transition)
    
    def compute_actualization_gradient(self) -> Optional[np.ndarray]:
        """Compute gradient toward actualization if telos is available.
        
        Returns:
            Gradient vector or None if telos not available
        """
        if self.telos is None:
            return None
        
        try:
            from ..telos import DevelopmentalAttractor
            
            # Create attractor
            attractor = DevelopmentalAttractor(self.telos)
            
            # Get current state
            current_state = self.coefficients.copy()
            
            # Compute gradient
            gradient = attractor.compute_gradient(current_state)
            
            return gradient
        except:
            return None
    
    def apply_teleological_pull(self, strength: float = 1.0) -> None:
        """Apply attractive force toward telos.
        
        Args:
            strength: Multiplier for pull strength
        """
        if self.telos is None:
            return
        
        try:
            from ..telos import DevelopmentalAttractor
            
            # Create attractor
            attractor = DevelopmentalAttractor(self.telos)
            
            # Get gradient
            gradient = self.compute_actualization_gradient()
            
            if gradient is not None:
                # Apply gradient to coefficients
                for i in range(min(len(self.coefficients), len(gradient))):
                    self.coefficients[i] += gradient[i] * strength * 0.1
                    # Keep in reasonable range
                    self.coefficients[i] = max(-10.0, min(10.0, self.coefficients[i]))
                
                # Record event
                if self.ontogenetic_state is not None:
                    self.ontogenetic_state.record_event(
                        "teleological_pull",
                        {"strength": strength, "gradient_norm": float(np.linalg.norm(gradient))}
                    )
        except:
            pass
