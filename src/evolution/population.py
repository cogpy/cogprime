"""
Population Management for MOSES Evolution

This module provides population management capabilities for the evolutionary algorithm,
including population initialization, diversity maintenance, and generational tracking.
"""

import random
import copy
from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .moses_engine import Individual, Gene
from src.atomspace import Atom, Node, Link


class PopulationInitializer(ABC):
    """Abstract base class for population initialization strategies."""
    
    @abstractmethod
    def initialize(self, size: int, context: Dict[str, Any] = None) -> List[Individual]:
        """Initialize a population of the specified size."""
        pass


class RandomInitializer(PopulationInitializer):
    """Random population initializer."""
    
    def __init__(self, atom_pool: List[Atom], 
                 min_genes: int = 1, max_genes: int = 5):
        """Initialize with atom pool and gene count bounds.
        
        Args:
            atom_pool: Pool of atoms to create genes from
            min_genes: Minimum number of genes per individual
            max_genes: Maximum number of genes per individual
        """
        self.atom_pool = atom_pool
        self.min_genes = min_genes
        self.max_genes = max_genes
    
    def initialize(self, size: int, context: Dict[str, Any] = None) -> List[Individual]:
        """Create random population."""
        population = []
        
        for _ in range(size):
            individual = Individual()
            
            # Random number of genes
            num_genes = random.randint(self.min_genes, 
                                     min(self.max_genes, len(self.atom_pool)))
            
            # Select random atoms
            selected_atoms = random.sample(self.atom_pool, num_genes)
            
            for atom in selected_atoms:
                gene = Gene(
                    atom=atom,
                    weight=random.random(),
                    active=random.random() > 0.2  # 80% active
                )
                individual.add_gene(gene)
            
            population.append(individual)
        
        return population


class SeededInitializer(PopulationInitializer):
    """Population initializer using seed patterns."""
    
    def __init__(self, seed_patterns: List[List[Atom]], 
                 variation_rate: float = 0.3):
        """Initialize with seed patterns.
        
        Args:
            seed_patterns: List of atom patterns to use as seeds
            variation_rate: Rate of variation from seed patterns
        """
        self.seed_patterns = seed_patterns
        self.variation_rate = variation_rate
    
    def initialize(self, size: int, context: Dict[str, Any] = None) -> List[Individual]:
        """Create population from seed patterns with variations."""
        population = []
        
        for i in range(size):
            # Select seed pattern cyclically
            pattern = self.seed_patterns[i % len(self.seed_patterns)]
            individual = Individual()
            
            for atom in pattern:
                # Add gene with possible variation
                weight = random.random()
                active = True
                
                # Apply variation
                if random.random() < self.variation_rate:
                    weight *= random.uniform(0.5, 1.5)
                    active = random.random() > 0.1
                
                gene = Gene(atom=atom, weight=weight, active=active)
                individual.add_gene(gene)
            
            population.append(individual)
        
        return population


@dataclass
class PopulationStats:
    """Statistics about a population."""
    size: int
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity: float
    active_genes_avg: float
    fitness_variance: float


class Population:
    """Manages a population of individuals for evolution."""
    
    def __init__(self, individuals: List[Individual] = None,
                 initializer: PopulationInitializer = None):
        """Initialize population.
        
        Args:
            individuals: Initial list of individuals
            initializer: Population initializer for creating new populations
        """
        self.individuals = individuals or []
        self.initializer = initializer
        self.generation = 0
        self.history: List[PopulationStats] = []
        
        # Population parameters
        self.diversity_threshold = 0.1  # Minimum diversity to maintain
        self.fitness_stagnation_limit = 10  # Generations before considering stagnant
        
    def __len__(self) -> int:
        """Return population size."""
        return len(self.individuals)
    
    def __iter__(self) -> Iterator[Individual]:
        """Iterate over individuals."""
        return iter(self.individuals)
    
    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]
    
    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the population."""
        individual.generation = self.generation
        self.individuals.append(individual)
    
    def remove_individual(self, index: int) -> Individual:
        """Remove and return individual at index."""
        if 0 <= index < len(self.individuals):
            return self.individuals.pop(index)
        raise IndexError("Individual index out of range")
    
    def sort_by_fitness(self, reverse: bool = True) -> None:
        """Sort population by fitness.
        
        Args:
            reverse: If True, sort in descending order (best first)
        """
        self.individuals.sort(key=lambda ind: ind.fitness or 0.0, reverse=reverse)
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """Get the best n individuals.
        
        Args:
            n: Number of best individuals to return
            
        Returns:
            List of best individuals
        """
        self.sort_by_fitness(reverse=True)
        return self.individuals[:n]
    
    def get_worst(self, n: int = 1) -> List[Individual]:
        """Get the worst n individuals.
        
        Args:
            n: Number of worst individuals to return
            
        Returns:
            List of worst individuals
        """
        self.sort_by_fitness(reverse=False)
        return self.individuals[:n]
    
    def calculate_stats(self) -> PopulationStats:
        """Calculate statistics for current population."""
        if not self.individuals:
            return PopulationStats(
                size=0, generation=self.generation, best_fitness=0.0,
                average_fitness=0.0, worst_fitness=0.0, diversity=0.0,
                active_genes_avg=0.0, fitness_variance=0.0
            )
        
        fitnesses = [ind.fitness or 0.0 for ind in self.individuals]
        active_genes_counts = [len(ind.get_active_genes()) for ind in self.individuals]
        
        avg_fitness = sum(fitnesses) / len(fitnesses)
        fitness_variance = sum((f - avg_fitness)**2 for f in fitnesses) / len(fitnesses)
        
        return PopulationStats(
            size=len(self.individuals),
            generation=self.generation,
            best_fitness=max(fitnesses) if fitnesses else 0.0,
            average_fitness=avg_fitness,
            worst_fitness=min(fitnesses) if fitnesses else 0.0,
            diversity=self._calculate_diversity(),
            active_genes_avg=sum(active_genes_counts) / len(active_genes_counts),
            fitness_variance=fitness_variance
        )
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.individuals) <= 1:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                distance = self._individual_distance(
                    self.individuals[i], 
                    self.individuals[j]
                )
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _individual_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate distance between two individuals."""
        genes1 = ind1.get_active_genes()
        genes2 = ind2.get_active_genes()
        
        if not genes1 and not genes2:
            return 0.0
        
        # Hamming-like distance based on atom types and weights
        all_atom_types = set()
        all_atom_types.update(gene.atom.atom_type for gene in genes1)
        all_atom_types.update(gene.atom.atom_type for gene in genes2)
        
        distance = 0.0
        for atom_type in all_atom_types:
            weight1 = sum(gene.weight for gene in genes1 
                         if gene.atom.atom_type == atom_type)
            weight2 = sum(gene.weight for gene in genes2 
                         if gene.atom.atom_type == atom_type)
            distance += abs(weight1 - weight2)
        
        return distance / len(all_atom_types) if all_atom_types else 0.0
    
    def maintain_diversity(self, target_diversity: float = None) -> int:
        """Maintain population diversity by replacing similar individuals.
        
        Args:
            target_diversity: Target diversity level (uses default if None)
            
        Returns:
            Number of individuals replaced
        """
        if target_diversity is None:
            target_diversity = self.diversity_threshold
        
        current_stats = self.calculate_stats()
        
        if current_stats.diversity >= target_diversity:
            return 0  # Diversity is sufficient
        
        # Find most similar pairs and replace the worse one
        replaced = 0
        similarity_threshold = 0.1  # Very similar individuals
        
        i = 0
        while i < len(self.individuals) - 1:
            j = i + 1
            while j < len(self.individuals):
                distance = self._individual_distance(
                    self.individuals[i], 
                    self.individuals[j]
                )
                
                if distance < similarity_threshold:
                    # Replace the worse individual with a new random one
                    if (self.individuals[i].fitness or 0.0) < (self.individuals[j].fitness or 0.0):
                        worse_idx = i
                    else:
                        worse_idx = j
                    
                    # Generate new individual if initializer available
                    if self.initializer:
                        new_individuals = self.initializer.initialize(1)
                        if new_individuals:
                            self.individuals[worse_idx] = new_individuals[0]
                            replaced += 1
                    
                    # Remove the j individual to avoid double replacement
                    if worse_idx == j:
                        j += 1
                    else:
                        break  # i was replaced, move to next i
                else:
                    j += 1
            i += 1
        
        return replaced
    
    def detect_stagnation(self, generations: int = None) -> bool:
        """Detect if population evolution has stagnated.
        
        Args:
            generations: Number of generations to check (uses default if None)
            
        Returns:
            True if evolution appears stagnant
        """
        if generations is None:
            generations = self.fitness_stagnation_limit
        
        if len(self.history) < generations:
            return False
        
        # Check if best fitness has improved in recent generations
        recent_best = [stats.best_fitness for stats in self.history[-generations:]]
        
        # Calculate improvement rate
        if len(recent_best) > 1:
            improvement = recent_best[-1] - recent_best[0]
            return improvement < 0.001  # Very small improvement
        
        return False
    
    def introduce_immigrants(self, count: int) -> int:
        """Introduce new random individuals (immigrants) to the population.
        
        Args:
            count: Number of immigrants to introduce
            
        Returns:
            Number of immigrants actually introduced
        """
        if not self.initializer:
            return 0
        
        # Replace worst individuals with immigrants
        self.sort_by_fitness(reverse=False)  # Worst first
        
        immigrants = self.initializer.initialize(count)
        actual_count = min(count, len(self.individuals))
        
        for i in range(actual_count):
            self.individuals[i] = immigrants[i % len(immigrants)]
        
        return actual_count
    
    def next_generation(self) -> None:
        """Advance to the next generation."""
        self.generation += 1
        
        # Update generation for all individuals
        for individual in self.individuals:
            individual.age += 1
            individual.generation = self.generation
        
        # Record statistics
        stats = self.calculate_stats()
        self.history.append(stats)
        
        # Automatic diversity maintenance
        if stats.diversity < self.diversity_threshold:
            self.maintain_diversity()
        
        # Check for stagnation and introduce immigrants if needed
        if self.detect_stagnation():
            immigrant_count = max(1, len(self.individuals) // 20)  # 5% immigrants
            self.introduce_immigrants(immigrant_count)
    
    def get_species_clusters(self, similarity_threshold: float = 0.3) -> List[List[Individual]]:
        """Group individuals into species based on similarity.
        
        Args:
            similarity_threshold: Threshold for species clustering
            
        Returns:
            List of species clusters
        """
        if not self.individuals:
            return []
        
        species = []
        unassigned = self.individuals.copy()
        
        while unassigned:
            # Start new species with first unassigned individual
            representative = unassigned.pop(0)
            current_species = [representative]
            
            # Find all individuals similar to representative
            i = 0
            while i < len(unassigned):
                distance = self._individual_distance(representative, unassigned[i])
                if distance <= similarity_threshold:
                    current_species.append(unassigned.pop(i))
                else:
                    i += 1
            
            species.append(current_species)
        
        return species
    
    def balance_species(self, target_species_size: int = None) -> None:
        """Balance species sizes to maintain diversity.
        
        Args:
            target_species_size: Target size for each species
        """
        species = self.get_species_clusters()
        
        if not species:
            return
        
        if target_species_size is None:
            target_species_size = len(self.individuals) // len(species)
        
        new_population = []
        
        for species_individuals in species:
            if len(species_individuals) > target_species_size:
                # Trim species - keep best individuals
                species_individuals.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)
                new_population.extend(species_individuals[:target_species_size])
            else:
                # Add all individuals from this species
                new_population.extend(species_individuals)
        
        self.individuals = new_population