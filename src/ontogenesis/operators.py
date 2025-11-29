"""
Differential operators for kernel ontogenesis.

This module implements the core operations for kernel self-generation,
self-optimization, and self-reproduction using differential calculus.
"""

from enum import Enum
from typing import Optional
import numpy as np
import uuid

from .kernel import (
    OntogeneticKernel,
    GeneratedKernel,
    GripMetrics,
    OntogeneticState,
    DevelopmentStage,
)
from .genome import (
    KernelGenome,
    CoefficientGene,
    crossover_genomes,
    mutate_genome,
)


class ReproductionMethod(Enum):
    """Methods for kernel reproduction."""

    CROSSOVER = "crossover"
    MUTATION = "mutation"
    CLONING = "cloning"


def initialize_ontogenetic_kernel(kernel: GeneratedKernel) -> OntogeneticKernel:
    """Convert a GeneratedKernel to an OntogeneticKernel.

    Args:
        kernel: Base generated kernel

    Returns:
        Ontogenetic kernel with genome and state
    """
    # Create genome from kernel coefficients
    genes = [
        CoefficientGene(index=i, value=coeff) for i, coeff in enumerate(kernel.coefficients)
    ]

    genome = KernelGenome(
        id=f"kernel_{uuid.uuid4().hex[:8]}",
        generation=0,
        lineage=[],
        genes=genes,
        fitness=kernel.grip_metrics.total_grip,
        age=0,
    )

    # Create ontogenetic state
    state = OntogeneticState(stage=DevelopmentStage.EMBRYONIC, maturity=0.0)

    # Create ontogenetic kernel
    onto_kernel = OntogeneticKernel(
        order=kernel.order,
        coefficients=kernel.coefficients.copy(),
        domain_spec=kernel.domain_spec,
        grip_metrics=kernel.grip_metrics,
        trees=kernel.trees.copy() if kernel.trees else [],
        genome=genome,
        ontogenetic_state=state,
    )

    return onto_kernel


def self_generate(parent: OntogeneticKernel) -> OntogeneticKernel:
    """Generate offspring through recursive self-composition.

    Applies the chain rule: (f∘f)' = f'(f(x)) · f'(x)

    Args:
        parent: Parent kernel

    Returns:
        Offspring kernel
    """
    # Create sample point for evaluation
    x = np.ones(len(parent.coefficients))

    # Compute f(x) and f'(x)
    f_x = parent.evaluate(x)
    f_prime = parent.differentiate(x)

    # Apply chain rule: compose kernel with itself
    # New coefficients are approximately f'(f(x)) * f'(x)
    composed_coefficients = f_prime * np.abs(f_x)

    # Normalize to prevent explosion
    if np.max(np.abs(composed_coefficients)) > 0:
        composed_coefficients = composed_coefficients / np.max(np.abs(composed_coefficients))

    # Create new grip metrics (inherit with slight variation)
    new_grip = GripMetrics(
        contact=parent.grip_metrics.contact * 0.95,
        coverage=parent.grip_metrics.coverage * 0.95,
        efficiency=parent.grip_metrics.efficiency * 1.05,
        stability=parent.grip_metrics.stability * 0.98,
    )

    # Create offspring genome with unique ID
    offspring_id = f"{parent.genome.id}_gen_{uuid.uuid4().hex[:8]}"
    new_genes = [
        CoefficientGene(index=i, value=coeff)
        for i, coeff in enumerate(composed_coefficients)
    ]

    offspring_genome = KernelGenome(
        id=offspring_id,
        generation=parent.genome.generation + 1,
        lineage=[parent.genome.id],
        genes=new_genes,
        fitness=new_grip.total_grip,
        age=0,
    )

    # Create offspring
    offspring = OntogeneticKernel(
        order=parent.order,
        coefficients=composed_coefficients,
        domain_spec=parent.domain_spec,
        grip_metrics=new_grip,
        trees=parent.trees.copy() if parent.trees else [],
        genome=offspring_genome,
        ontogenetic_state=OntogeneticState(stage=DevelopmentStage.EMBRYONIC, maturity=0.0),
    )

    # Record event in parent
    if parent.ontogenetic_state:
        parent.ontogenetic_state.record_event(
            "self_generation", {"offspring_id": offspring.genome.id}
        )

    return offspring


def self_optimize(kernel: OntogeneticKernel, iterations: int = 10) -> OntogeneticKernel:
    """Optimize kernel through iterative grip improvement.

    Args:
        kernel: Kernel to optimize
        iterations: Number of optimization iterations

    Returns:
        Optimized kernel
    """
    current_coeffs = kernel.coefficients.copy()
    current_grip = kernel.grip_metrics.total_grip
    best_coeffs = current_coeffs.copy()
    best_grip = current_grip

    # Simple gradient ascent for grip optimization
    learning_rate = 0.01

    for i in range(iterations):
        # Compute gradient approximation
        gradient = np.random.randn(len(current_coeffs)) * learning_rate

        # Apply gradient
        new_coeffs = current_coeffs + gradient

        # Evaluate new grip (simple metric: prefer smaller magnitude with more spread)
        spread = np.std(new_coeffs)
        magnitude = np.mean(np.abs(new_coeffs))
        new_grip_score = spread / (magnitude + 1e-6)

        # Update if improved
        if new_grip_score > best_grip:
            best_coeffs = new_coeffs.copy()
            best_grip = new_grip_score
            current_coeffs = new_coeffs

        # Update maturity
        kernel.ontogenetic_state.maturity = min(1.0, (i + 1) / iterations)

    # Update kernel with optimized coefficients
    kernel.coefficients = best_coeffs

    # Update grip metrics
    kernel.grip_metrics.contact = min(1.0, kernel.grip_metrics.contact * 1.1)
    kernel.grip_metrics.stability = min(1.0, kernel.grip_metrics.stability * 1.05)

    # Update genome
    new_genes = [CoefficientGene(index=i, value=coeff) for i, coeff in enumerate(best_coeffs)]
    kernel.genome.genes = new_genes
    kernel.genome.update_fitness(kernel.grip_metrics.total_grip)

    # Record optimization event
    kernel.ontogenetic_state.record_event(
        "self_optimization", {"iterations": iterations, "final_grip": best_grip}
    )

    # Advance stage if mature enough
    kernel.advance_stage()

    return kernel


def self_reproduce(
    parent1: OntogeneticKernel,
    parent2: OntogeneticKernel,
    method: ReproductionMethod = ReproductionMethod.CROSSOVER,
    mutation_rate: float = 0.1,
) -> tuple[OntogeneticKernel, OntogeneticKernel]:
    """Combine two kernels to create offspring.

    Args:
        parent1: First parent kernel
        parent2: Second parent kernel
        method: Reproduction method
        mutation_rate: Mutation probability

    Returns:
        Tuple of two offspring kernels
    """
    if method == ReproductionMethod.CLONING:
        # Simple cloning of first parent
        offspring1 = _clone_kernel(parent1)
        offspring2 = _clone_kernel(parent2)
        return offspring1, offspring2

    elif method == ReproductionMethod.MUTATION:
        # Mutate both parents
        mutated_genome1 = mutate_genome(parent1.genome, mutation_rate)
        mutated_genome2 = mutate_genome(parent2.genome, mutation_rate)
        offspring1 = _genome_to_kernel(mutated_genome1, parent1)
        offspring2 = _genome_to_kernel(mutated_genome2, parent2)
        return offspring1, offspring2

    else:  # CROSSOVER
        # Perform genetic crossover
        offspring_genome1, offspring_genome2 = crossover_genomes(parent1.genome, parent2.genome)

        # Apply mutation
        if np.random.random() < mutation_rate:
            offspring_genome1 = mutate_genome(offspring_genome1, mutation_rate)
        if np.random.random() < mutation_rate:
            offspring_genome2 = mutate_genome(offspring_genome2, mutation_rate)

        # Create kernel objects
        offspring1 = _genome_to_kernel(offspring_genome1, parent1)
        offspring2 = _genome_to_kernel(offspring_genome2, parent2)

        return offspring1, offspring2


def _clone_kernel(kernel: OntogeneticKernel) -> OntogeneticKernel:
    """Create a clone of a kernel."""
    cloned_genome = kernel.genome.clone()
    cloned_genome.id = f"{kernel.genome.id}_clone"

    return OntogeneticKernel(
        order=kernel.order,
        coefficients=kernel.coefficients.copy(),
        domain_spec=kernel.domain_spec,
        grip_metrics=GripMetrics(
            contact=kernel.grip_metrics.contact,
            coverage=kernel.grip_metrics.coverage,
            efficiency=kernel.grip_metrics.efficiency,
            stability=kernel.grip_metrics.stability,
        ),
        trees=kernel.trees.copy() if kernel.trees else [],
        genome=cloned_genome,
        ontogenetic_state=OntogeneticState(stage=DevelopmentStage.EMBRYONIC, maturity=0.0),
    )


def _genome_to_kernel(
    genome: KernelGenome, template: OntogeneticKernel
) -> OntogeneticKernel:
    """Create a kernel from a genome using a template kernel."""
    coefficients = genome.get_coefficients()

    # Estimate grip based on coefficient properties
    spread = np.std(coefficients)
    magnitude = np.mean(np.abs(coefficients))
    grip_score = spread / (magnitude + 1e-6)

    grip = GripMetrics(
        contact=min(1.0, grip_score * 0.8),
        coverage=min(1.0, grip_score * 0.7),
        efficiency=min(1.0, 1.0 / (magnitude + 1.0)),
        stability=min(1.0, 1.0 / (spread + 1.0)),
    )

    # Update genome fitness
    genome.update_fitness(grip.total_grip)

    return OntogeneticKernel(
        order=template.order,
        coefficients=coefficients,
        domain_spec=template.domain_spec,
        grip_metrics=grip,
        trees=template.trees.copy() if template.trees else [],
        genome=genome,
        ontogenetic_state=OntogeneticState(stage=DevelopmentStage.EMBRYONIC, maturity=0.0),
    )
