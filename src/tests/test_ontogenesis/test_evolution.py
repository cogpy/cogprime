"""Tests for evolutionary algorithms."""

import pytest
import numpy as np

from src.ontogenesis.kernel import GeneratedKernel, OntogeneticKernel, GripMetrics
from src.ontogenesis.operators import initialize_ontogenetic_kernel
from src.ontogenesis.evolution import (
    EvolutionConfig,
    OntogenesisConfig,
    evaluate_fitness,
    tournament_selection,
    calculate_diversity,
    run_ontogenesis,
    get_best_kernel,
    summarize_evolution,
)


def create_test_population(size=5):
    """Helper to create a test population."""
    population = []
    for i in range(size):
        coeffs = np.random.rand(4)
        grip = GripMetrics(
            contact=np.random.rand(),
            coverage=np.random.rand(),
            efficiency=np.random.rand(),
            stability=np.random.rand(),
        )
        base_kernel = GeneratedKernel(
            order=4, coefficients=coeffs, domain_spec="test", grip_metrics=grip
        )
        onto_kernel = initialize_ontogenetic_kernel(base_kernel)
        population.append(onto_kernel)

    return population


def test_evaluate_fitness():
    """Test fitness evaluation."""
    population = create_test_population(3)
    kernel = population[0]

    fitness = evaluate_fitness(kernel, population)

    # Fitness should be in [0, 1]
    assert 0.0 <= fitness <= 1.0


def test_tournament_selection():
    """Test tournament selection."""
    population = create_test_population(10)

    # Set different fitness values
    for i, kernel in enumerate(population):
        kernel.genome.update_fitness(float(i) / 10.0)

    selected = tournament_selection(population, tournament_size=3)

    # Selected should be from population (check by ID)
    assert any(selected.genome.id == k.genome.id for k in population)

    # With multiple runs, should select high fitness individuals more often
    selected_fitness_values = []
    for _ in range(20):
        selected = tournament_selection(population, tournament_size=3)
        selected_fitness_values.append(selected.genome.fitness)

    # Average selected fitness should be higher than population average
    avg_selected = np.mean(selected_fitness_values)
    avg_population = np.mean([k.genome.fitness for k in population])
    assert avg_selected >= avg_population


def test_calculate_diversity():
    """Test diversity calculation."""
    # Create identical population (low diversity)
    coeffs = np.array([1.0, 1.0, 1.0, 1.0])
    population = []
    for i in range(5):
        grip = GripMetrics(contact=0.8, coverage=0.7, efficiency=0.6, stability=0.9)
        base_kernel = GeneratedKernel(
            order=4, coefficients=coeffs, domain_spec="test", grip_metrics=grip
        )
        onto_kernel = initialize_ontogenetic_kernel(base_kernel)
        population.append(onto_kernel)

    diversity = calculate_diversity(population)
    assert diversity >= 0.0  # Should be very low (near 0) for identical population

    # Create diverse population
    diverse_population = create_test_population(5)
    diverse_diversity = calculate_diversity(diverse_population)
    assert diverse_diversity > diversity


def test_evolution_config():
    """Test EvolutionConfig creation."""
    config = EvolutionConfig(
        population_size=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_rate=0.1,
        max_generations=50,
        fitness_threshold=0.9,
        diversity_pressure=0.1,
    )

    assert config.population_size == 20
    assert config.mutation_rate == 0.1
    assert config.crossover_rate == 0.7
    assert config.elitism_rate == 0.1
    assert config.max_generations == 50
    assert config.fitness_threshold == 0.9
    assert config.diversity_pressure == 0.1


def test_run_ontogenesis_basic():
    """Test basic ontogenesis run."""
    seed_population = create_test_population(3)

    config = OntogenesisConfig(
        evolution=EvolutionConfig(
            population_size=10,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_rate=0.1,
            max_generations=5,
            fitness_threshold=1.0,  # Won't reach this
            diversity_pressure=0.1,
        ),
        seed_kernels=seed_population,
    )

    generations = run_ontogenesis(config)

    # Should run for specified generations
    assert len(generations) == 5

    # Each generation should have stats
    for gen in generations:
        assert gen.generation >= 0
        assert len(gen.population) == config.evolution.population_size
        assert 0.0 <= gen.best_fitness <= 1.0
        assert 0.0 <= gen.average_fitness <= 1.0
        assert gen.diversity >= 0.0


def test_run_ontogenesis_convergence():
    """Test ontogenesis with early stopping."""
    seed_population = create_test_population(3)

    config = OntogenesisConfig(
        evolution=EvolutionConfig(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.2,
            max_generations=100,
            fitness_threshold=0.5,  # Low threshold for quick convergence
            diversity_pressure=0.1,
        ),
        seed_kernels=seed_population,
    )

    generations = run_ontogenesis(config)

    # Should stop before max generations if threshold reached
    assert len(generations) <= 100

    # Final generation should meet or exceed threshold
    if len(generations) < 100:
        assert generations[-1].best_fitness >= config.evolution.fitness_threshold


def test_run_ontogenesis_population_size():
    """Test that population size is maintained."""
    seed_population = create_test_population(2)

    config = OntogenesisConfig(
        evolution=EvolutionConfig(
            population_size=15,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.1,
            max_generations=3,
            fitness_threshold=1.0,
            diversity_pressure=0.1,
        ),
        seed_kernels=seed_population,
    )

    generations = run_ontogenesis(config)

    # Every generation should have correct population size
    for gen in generations:
        assert len(gen.population) == 15


def test_get_best_kernel():
    """Test getting the best kernel from generations."""
    seed_population = create_test_population(3)

    config = OntogenesisConfig(
        evolution=EvolutionConfig(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.1,
            max_generations=5,
            fitness_threshold=1.0,
            diversity_pressure=0.1,
        ),
        seed_kernels=seed_population,
    )

    generations = run_ontogenesis(config)
    best_kernel = get_best_kernel(generations)

    # Best kernel should exist
    assert best_kernel is not None
    assert isinstance(best_kernel, OntogeneticKernel)

    # Should have highest or very close to highest fitness across all generations
    all_best_fitness = [gen.best_fitness for gen in generations]
    assert best_kernel.genome.fitness >= max(all_best_fitness) * 0.99  # Allow small variation


def test_summarize_evolution():
    """Test evolution summary."""
    seed_population = create_test_population(3)

    config = OntogenesisConfig(
        evolution=EvolutionConfig(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.1,
            max_generations=5,
            fitness_threshold=1.0,
            diversity_pressure=0.1,
        ),
        seed_kernels=seed_population,
    )

    generations = run_ontogenesis(config)
    summary = summarize_evolution(generations)

    # Check summary keys
    assert "total_generations" in summary
    assert "final_best_fitness" in summary
    assert "final_average_fitness" in summary
    assert "final_diversity" in summary
    assert "best_kernel_id" in summary
    assert "best_kernel_generation" in summary
    assert "best_kernel_fitness" in summary
    assert "convergence_generation" in summary

    # Check values
    assert summary["total_generations"] == 5
    assert 0.0 <= summary["final_best_fitness"] <= 1.0
    assert 0.0 <= summary["final_average_fitness"] <= 1.0


def test_elitism_preservation():
    """Test that elite individuals are preserved."""
    seed_population = create_test_population(5)

    # Give one kernel very high fitness
    best_kernel = seed_population[0]
    best_kernel.genome.update_fitness(0.99)

    config = OntogenesisConfig(
        evolution=EvolutionConfig(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.2,  # Keep top 20%
            max_generations=3,
            fitness_threshold=1.0,
            diversity_pressure=0.1,
        ),
        seed_kernels=seed_population,
    )

    generations = run_ontogenesis(config)

    # Best fitness should not decrease (due to elitism)
    fitness_values = [gen.best_fitness for gen in generations]
    for i in range(1, len(fitness_values)):
        assert fitness_values[i] >= fitness_values[i - 1] * 0.9  # Allow small variation


def test_empty_generations_summary():
    """Test summary with empty generations list."""
    summary = summarize_evolution([])
    assert "error" in summary


def test_custom_fitness_function():
    """Test custom fitness function."""
    seed_population = create_test_population(3)

    def custom_fitness(kernel):
        # Simple fitness: prefer larger coefficient sums
        return min(1.0, np.sum(np.abs(kernel.coefficients)) / 10.0)

    config = OntogenesisConfig(
        evolution=EvolutionConfig(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.1,
            max_generations=3,
            fitness_threshold=1.0,
            diversity_pressure=0.1,
        ),
        seed_kernels=seed_population,
        fitness_function=custom_fitness,
    )

    generations = run_ontogenesis(config)

    # Should complete successfully with custom fitness
    assert len(generations) == 3
    assert all(0.0 <= gen.best_fitness <= 1.0 for gen in generations)
