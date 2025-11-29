"""Tests for differential operators."""

import pytest
import numpy as np

from src.ontogenesis.kernel import (
    GeneratedKernel,
    OntogeneticKernel,
    GripMetrics,
    DevelopmentStage,
)
from src.ontogenesis.operators import (
    initialize_ontogenetic_kernel,
    self_generate,
    self_optimize,
    self_reproduce,
    ReproductionMethod,
)


def create_test_kernel(coeffs=None):
    """Helper to create a test kernel."""
    if coeffs is None:
        coeffs = np.array([1.0, 0.5, 0.25, 0.125])

    grip = GripMetrics(contact=0.8, coverage=0.7, efficiency=0.6, stability=0.9)

    return GeneratedKernel(
        order=len(coeffs),
        coefficients=coeffs,
        domain_spec="test_domain",
        grip_metrics=grip,
        trees=["f", "f'*f"],
    )


def test_initialize_ontogenetic_kernel():
    """Test initialization of ontogenetic kernel."""
    base_kernel = create_test_kernel()
    onto_kernel = initialize_ontogenetic_kernel(base_kernel)

    assert isinstance(onto_kernel, OntogeneticKernel)
    assert onto_kernel.genome is not None
    assert onto_kernel.ontogenetic_state is not None
    assert onto_kernel.ontogenetic_state.stage == DevelopmentStage.EMBRYONIC
    assert len(onto_kernel.genome.genes) == len(base_kernel.coefficients)


def test_self_generate():
    """Test self-generation of kernel."""
    base_kernel = create_test_kernel()
    parent = initialize_ontogenetic_kernel(base_kernel)

    offspring = self_generate(parent)

    # Offspring should be different
    assert offspring is not parent
    assert offspring.genome.id != parent.genome.id
    assert offspring.genome.generation == parent.genome.generation + 1
    assert offspring.genome.lineage == [parent.genome.id]
    assert offspring.ontogenetic_state.stage == DevelopmentStage.EMBRYONIC

    # Should have recorded event
    assert len(parent.ontogenetic_state.development_history) > 0


def test_self_optimize():
    """Test self-optimization of kernel."""
    base_kernel = create_test_kernel()
    kernel = initialize_ontogenetic_kernel(base_kernel)

    initial_grip = kernel.grip_metrics.total_grip
    initial_maturity = kernel.ontogenetic_state.maturity

    optimized = self_optimize(kernel, iterations=5)

    # Should be same kernel object, modified
    assert optimized is kernel

    # Maturity should increase
    assert kernel.ontogenetic_state.maturity > initial_maturity

    # Should have recorded optimization event
    events = kernel.ontogenetic_state.development_history
    optimization_events = [e for e in events if e["type"] == "self_optimization"]
    assert len(optimization_events) > 0


def test_self_optimize_stage_advancement():
    """Test that optimization advances development stage."""
    base_kernel = create_test_kernel()
    kernel = initialize_ontogenetic_kernel(base_kernel)

    assert kernel.ontogenetic_state.stage == DevelopmentStage.EMBRYONIC

    # Optimize to full maturity
    self_optimize(kernel, iterations=20)

    # Should advance to at least juvenile stage
    assert kernel.ontogenetic_state.stage != DevelopmentStage.EMBRYONIC


def test_self_reproduce_crossover():
    """Test reproduction with crossover."""
    parent1 = initialize_ontogenetic_kernel(create_test_kernel(np.array([1.0, 1.0, 1.0, 1.0])))
    parent2 = initialize_ontogenetic_kernel(create_test_kernel(np.array([2.0, 2.0, 2.0, 2.0])))

    offspring1, offspring2 = self_reproduce(
        parent1, parent2, method=ReproductionMethod.CROSSOVER, mutation_rate=0.0
    )

    # Check offspring are different objects
    assert offspring1 is not parent1
    assert offspring2 is not parent2
    assert offspring1 is not offspring2

    # Check lineage
    assert len(offspring1.genome.lineage) == 2
    assert parent1.genome.id in offspring1.genome.lineage
    assert parent2.genome.id in offspring1.genome.lineage


def test_self_reproduce_mutation():
    """Test reproduction with mutation."""
    parent1 = initialize_ontogenetic_kernel(create_test_kernel())
    parent2 = initialize_ontogenetic_kernel(create_test_kernel())

    offspring1, offspring2 = self_reproduce(
        parent1, parent2, method=ReproductionMethod.MUTATION, mutation_rate=1.0
    )

    # Offspring should be different from parents
    assert offspring1 is not parent1
    assert offspring2 is not parent2

    # Coefficients should be mutated
    assert not np.allclose(offspring1.coefficients, parent1.coefficients)
    assert not np.allclose(offspring2.coefficients, parent2.coefficients)


def test_self_reproduce_cloning():
    """Test reproduction with cloning."""
    parent1 = initialize_ontogenetic_kernel(create_test_kernel())
    parent2 = initialize_ontogenetic_kernel(create_test_kernel())

    offspring1, offspring2 = self_reproduce(parent1, parent2, method=ReproductionMethod.CLONING)

    # Offspring should be clones (similar coefficients)
    assert offspring1 is not parent1
    assert offspring2 is not parent2
    assert np.allclose(offspring1.coefficients, parent1.coefficients)
    assert np.allclose(offspring2.coefficients, parent2.coefficients)


def test_multiple_generations():
    """Test multiple generations of self-generation."""
    base_kernel = create_test_kernel()
    current = initialize_ontogenetic_kernel(base_kernel)

    lineage = [current]

    # Generate 5 generations
    for i in range(5):
        current = self_generate(current)
        lineage.append(current)

    # Check generations
    for i, kernel in enumerate(lineage):
        assert kernel.genome.generation == i

    # Check lineage tracking
    last_kernel = lineage[-1]
    assert len(last_kernel.genome.lineage) == 1


def test_reproduction_preserves_domain():
    """Test that reproduction preserves domain specification."""
    parent1 = initialize_ontogenetic_kernel(create_test_kernel())
    parent2 = initialize_ontogenetic_kernel(create_test_kernel())

    offspring1, offspring2 = self_reproduce(parent1, parent2)

    assert offspring1.domain_spec == parent1.domain_spec
    assert offspring2.domain_spec == parent2.domain_spec


def test_generation_creates_unique_ids():
    """Test that generated kernels have unique IDs."""
    base_kernel = create_test_kernel()
    parent = initialize_ontogenetic_kernel(base_kernel)

    offspring1 = self_generate(parent)
    offspring2 = self_generate(parent)

    assert offspring1.genome.id != offspring2.genome.id
    assert offspring1.genome.id != parent.genome.id
    assert offspring2.genome.id != parent.genome.id
