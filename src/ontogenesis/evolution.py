"""
Evolutionary algorithms for kernel populations.

This module implements population-based evolution of ontogenetic kernels,
including selection, reproduction, and generational advancement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np

from .kernel import OntogeneticKernel, DevelopmentStage
from .genome import genetic_distance
from .operators import self_reproduce, ReproductionMethod


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""

    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    max_generations: int = 100
    fitness_threshold: float = 0.9
    diversity_pressure: float = 0.1


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    population: List[OntogeneticKernel]
    best_fitness: float
    average_fitness: float
    diversity: float
    best_individual: OntogeneticKernel

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "diversity": self.diversity,
            "best_id": self.best_individual.genome.id,
        }


@dataclass
class OntogenesisConfig:
    """Complete configuration for ontogenesis run."""

    evolution: EvolutionConfig
    seed_kernels: List[OntogeneticKernel]
    fitness_function: Optional[Callable[[OntogeneticKernel], float]] = None


def evaluate_fitness(kernel: OntogeneticKernel, population: List[OntogeneticKernel]) -> float:
    """Evaluate fitness of a kernel considering multiple factors.

    Fitness = grip * 0.4 + stability * 0.2 + efficiency * 0.2 + novelty * 0.1 + symmetry * 0.1

    Args:
        kernel: Kernel to evaluate
        population: Current population for novelty calculation

    Returns:
        Fitness score (0.0 to 1.0)
    """
    grip = kernel.grip_metrics.total_grip
    stability = kernel.grip_metrics.stability
    efficiency = kernel.grip_metrics.efficiency

    # Calculate novelty as average genetic distance from population
    if len(population) > 1:
        distances = [genetic_distance(kernel.genome, k.genome) for k in population if k.genome.id != kernel.genome.id]
        novelty = np.mean(distances) if distances else 0.0
        novelty = min(1.0, novelty / 10.0)  # Normalize
    else:
        novelty = 0.5

    # Symmetry score (based on coefficient symmetry)
    coeffs = kernel.coefficients
    if len(coeffs) > 1:
        # Measure how symmetric coefficients are
        reversed_coeffs = coeffs[::-1]
        symmetry = 1.0 - np.mean(np.abs(coeffs - reversed_coeffs)) / (np.mean(np.abs(coeffs)) + 1e-6)
        symmetry = max(0.0, min(1.0, symmetry))
    else:
        symmetry = 0.5

    # Weighted combination
    fitness = grip * 0.4 + stability * 0.2 + efficiency * 0.2 + novelty * 0.1 + symmetry * 0.1

    return min(1.0, max(0.0, fitness))


def tournament_selection(
    population: List[OntogeneticKernel], tournament_size: int = 3
) -> OntogeneticKernel:
    """Select an individual using tournament selection.

    Args:
        population: Population to select from
        tournament_size: Number of individuals in tournament

    Returns:
        Selected individual
    """
    # Randomly select tournament participants
    indices = np.random.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    tournament = [population[i] for i in indices]
    return max(tournament, key=lambda k: k.genome.fitness)


def calculate_diversity(population: List[OntogeneticKernel]) -> float:
    """Calculate genetic diversity of population.

    Args:
        population: Population to measure

    Returns:
        Diversity score (higher = more diverse)
    """
    if len(population) < 2:
        return 0.0

    # Calculate pairwise genetic distances
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = genetic_distance(population[i].genome, population[j].genome)
            distances.append(dist)

    return np.mean(distances) if distances else 0.0


def run_ontogenesis(config: OntogenesisConfig) -> List[GenerationStats]:
    """Run evolutionary algorithm for kernel population.

    Args:
        config: Ontogenesis configuration

    Returns:
        List of generation statistics
    """
    evo_config = config.evolution
    population = config.seed_kernels.copy()

    # Expand initial population to target size
    while len(population) < evo_config.population_size:
        parent = np.random.choice(population)
        offspring1, offspring2 = self_reproduce(parent, parent, method=ReproductionMethod.MUTATION)
        population.append(offspring1)
        if len(population) < evo_config.population_size:
            population.append(offspring2)

    generations_stats = []

    # Main evolution loop
    for generation in range(evo_config.max_generations):
        # Evaluate fitness for all individuals
        for kernel in population:
            if config.fitness_function:
                fitness = config.fitness_function(kernel)
            else:
                fitness = evaluate_fitness(kernel, population)
            kernel.genome.update_fitness(fitness)

        # Sort by fitness
        population.sort(key=lambda k: k.genome.fitness, reverse=True)

        # Calculate statistics
        best_kernel = population[0]
        best_fitness = best_kernel.genome.fitness
        average_fitness = np.mean([k.genome.fitness for k in population])
        diversity = calculate_diversity(population)

        # Store generation stats
        stats = GenerationStats(
            generation=generation,
            population=[k for k in population],  # Store copy
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            diversity=diversity,
            best_individual=best_kernel,
        )
        generations_stats.append(stats)

        # Check termination
        if best_fitness >= evo_config.fitness_threshold:
            break

        # Create next generation
        new_population = []

        # Elitism: keep best individuals
        elite_count = int(evo_config.population_size * evo_config.elitism_rate)
        new_population.extend(population[:elite_count])

        # Generate offspring
        while len(new_population) < evo_config.population_size:
            # Select parents
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Reproduce
            if np.random.random() < evo_config.crossover_rate:
                offspring1, offspring2 = self_reproduce(
                    parent1,
                    parent2,
                    method=ReproductionMethod.CROSSOVER,
                    mutation_rate=evo_config.mutation_rate,
                )
            else:
                offspring1, offspring2 = self_reproduce(
                    parent1, parent2, method=ReproductionMethod.MUTATION, mutation_rate=evo_config.mutation_rate
                )

            new_population.append(offspring1)
            if len(new_population) < evo_config.population_size:
                new_population.append(offspring2)

        # Update ages and stages
        for kernel in new_population:
            kernel.genome.increment_age()
            # Advance development stage based on age and fitness
            if kernel.genome.fitness > 0.7 and kernel.ontogenetic_state.maturity < 1.0:
                kernel.ontogenetic_state.maturity = min(
                    1.0, kernel.ontogenetic_state.maturity + 0.1
                )
                kernel.advance_stage()

        population = new_population

    return generations_stats


def get_best_kernel(generations: List[GenerationStats]) -> OntogeneticKernel:
    """Get the best kernel across all generations.

    Args:
        generations: List of generation statistics

    Returns:
        Best kernel found
    """
    if not generations:
        raise ValueError("No generations provided")

    best_generation = max(generations, key=lambda g: g.best_fitness)
    return best_generation.best_individual


def summarize_evolution(generations: List[GenerationStats]) -> dict:
    """Summarize evolution results.

    Args:
        generations: List of generation statistics

    Returns:
        Summary dictionary
    """
    if not generations:
        return {"error": "No generations to summarize"}

    final_gen = generations[-1]
    best_kernel = get_best_kernel(generations)

    return {
        "total_generations": len(generations),
        "final_best_fitness": final_gen.best_fitness,
        "final_average_fitness": final_gen.average_fitness,
        "final_diversity": final_gen.diversity,
        "best_kernel_id": best_kernel.genome.id,
        "best_kernel_generation": best_kernel.genome.generation,
        "best_kernel_fitness": best_kernel.genome.fitness,
        "convergence_generation": next(
            (i for i, g in enumerate(generations) if g.best_fitness > 0.8), len(generations)
        ),
    }
