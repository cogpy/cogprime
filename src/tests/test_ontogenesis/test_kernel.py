"""Tests for kernel data structures."""

import pytest
import numpy as np

from src.ontogenesis.kernel import (
    OntogeneticKernel,
    GeneratedKernel,
    GripMetrics,
    OntogeneticState,
    DevelopmentStage,
)


def test_grip_metrics():
    """Test GripMetrics calculation."""
    grip = GripMetrics(contact=0.8, coverage=0.7, efficiency=0.6, stability=0.9)

    total = grip.total_grip
    assert 0.0 <= total <= 1.0
    # Weighted: 0.8*0.4 + 0.7*0.3 + 0.9*0.2 + 0.6*0.1 = 0.32 + 0.21 + 0.18 + 0.06 = 0.77
    assert abs(total - 0.77) < 0.01


def test_ontogenetic_state():
    """Test OntogeneticState."""
    state = OntogeneticState(stage=DevelopmentStage.EMBRYONIC, maturity=0.0)

    assert state.stage == DevelopmentStage.EMBRYONIC
    assert state.maturity == 0.0
    assert len(state.development_history) == 0

    # Record event
    state.record_event("test_event", {"detail": "value"})
    assert len(state.development_history) == 1
    assert state.development_history[0]["type"] == "test_event"


def test_generated_kernel_creation():
    """Test GeneratedKernel creation."""
    coeffs = np.array([1.0, 0.5, 0.25, 0.125])
    grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

    kernel = GeneratedKernel(
        order=4, coefficients=coeffs, domain_spec="test_domain", grip_metrics=grip, trees=["f", "f'*f"]
    )

    assert kernel.order == 4
    assert len(kernel.coefficients) == 4
    assert kernel.domain_spec == "test_domain"
    assert len(kernel.trees) == 2


def test_generated_kernel_evaluate():
    """Test kernel evaluation."""
    coeffs = np.array([1.0, 2.0, 3.0])
    grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

    kernel = GeneratedKernel(order=3, coefficients=coeffs, domain_spec="test", grip_metrics=grip)

    x = np.array([1.0, 1.0, 1.0])
    result = kernel.evaluate(x)

    # Should be sum of coefficients * x = 1*1 + 2*1 + 3*1 = 6
    assert abs(result - 6.0) < 0.01


def test_generated_kernel_differentiate():
    """Test kernel differentiation."""
    coeffs = np.array([1.0, 2.0, 3.0])
    grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

    kernel = GeneratedKernel(order=3, coefficients=coeffs, domain_spec="test", grip_metrics=grip)

    x = np.array([1.0, 1.0, 1.0])
    derivative = kernel.differentiate(x)

    # Should return coefficients
    assert len(derivative) == 3
    assert np.allclose(derivative, coeffs)


def test_ontogenetic_kernel_initialization():
    """Test OntogeneticKernel initialization."""
    coeffs = np.array([1.0, 0.5, 0.25])
    grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

    kernel = OntogeneticKernel(
        order=3, coefficients=coeffs, domain_spec="test_domain", grip_metrics=grip
    )

    # Should auto-create genome and state
    assert kernel.genome is not None
    assert kernel.ontogenetic_state is not None
    assert kernel.ontogenetic_state.stage == DevelopmentStage.EMBRYONIC
    assert kernel.genome.generation == 0
    assert len(kernel.genome.genes) == 3


def test_ontogenetic_kernel_fitness():
    """Test fitness calculation."""
    coeffs = np.array([1.0, 0.5, 0.25])
    grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

    kernel = OntogeneticKernel(
        order=3, coefficients=coeffs, domain_spec="test_domain", grip_metrics=grip
    )

    fitness = kernel.get_fitness()
    assert 0.0 <= fitness <= 1.0
    assert abs(fitness - grip.total_grip) < 0.01


def test_ontogenetic_kernel_stage_advancement():
    """Test development stage advancement."""
    coeffs = np.array([1.0, 0.5, 0.25])
    grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

    kernel = OntogeneticKernel(
        order=3, coefficients=coeffs, domain_spec="test_domain", grip_metrics=grip
    )

    assert kernel.ontogenetic_state.stage == DevelopmentStage.EMBRYONIC

    # Advance to juvenile (requires maturity >= 0.3)
    kernel.ontogenetic_state.maturity = 0.4
    kernel.advance_stage()
    assert kernel.ontogenetic_state.stage == DevelopmentStage.JUVENILE

    # Advance to mature (requires maturity >= 0.7)
    kernel.ontogenetic_state.maturity = 0.8
    kernel.advance_stage()
    assert kernel.ontogenetic_state.stage == DevelopmentStage.MATURE

    # Advance to senescent (requires maturity >= 0.95)
    kernel.ontogenetic_state.maturity = 0.96
    kernel.advance_stage()
    assert kernel.ontogenetic_state.stage == DevelopmentStage.SENESCENT


def test_kernel_validation():
    """Test kernel validation."""
    grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

    # Empty coefficients should raise error
    with pytest.raises(ValueError):
        GeneratedKernel(order=3, coefficients=np.array([]), domain_spec="test", grip_metrics=grip)

    # Order < 1 should raise error
    with pytest.raises(ValueError):
        GeneratedKernel(
            order=0, coefficients=np.array([1.0]), domain_spec="test", grip_metrics=grip
        )
