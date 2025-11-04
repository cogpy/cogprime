"""
MOSES Evolution Engine - Meta-Optimizing Semantic Evolutionary Search

This module implements the core MOSES evolutionary algorithm inspired by OpenCog's
MOSES system, adapted for cognitive structure evolution in the CogPrime framework.

MOSES combines evolutionary programming with program learning, using semantic
awareness to guide evolution of cognitive patterns and behaviors.
"""

import random
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import copy

# Import AtomSpace components
from src.atomspace import AtomSpace, Atom, Node, Link, TruthValue, AttentionValue


class EvolutionaryOperator(ABC):
    """Abstract base class for evolutionary operators."""
    
    @abstractmethod
    def apply(self, individual: 'Individual', context: Dict[str, Any] = None) -> 'Individual':
        """Apply the evolutionary operator to an individual."""
        pass
    
    @abstractmethod
    def get_probability(self) -> float:
        """Get the probability of applying this operator."""
        pass


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""
    
    @abstractmethod
    def evaluate(self, individual: 'Individual', context: Dict[str, Any] = None) -> float:
        """Evaluate the fitness of an individual."""
        pass


@dataclass
class Gene:
    """Represents a gene in the evolutionary algorithm."""
    atom: Atom
    weight: float = 1.0
    active: bool = True
    
    def copy(self) -> 'Gene':
        """Create a copy of this gene."""
        return Gene(
            atom=copy.deepcopy(self.atom),
            weight=self.weight,
            active=self.active
        )


class Individual:
    """Represents an individual in the evolutionary population."""
    
    def __init__(self, genes: List[Gene] = None):
        """Initialize an individual with genes.
        
        Args:
            genes: List of genes that make up this individual
        """
        self.genes = genes or []
        self.fitness: Optional[float] = None
        self.age = 0
        self.generation = 0
        self.metadata: Dict[str, Any] = {}
    
    def add_gene(self, gene: Gene) -> None:
        """Add a gene to this individual."""
        self.genes.append(gene)
    
    def remove_gene(self, index: int) -> Gene:
        """Remove and return a gene at the specified index."""
        if 0 <= index < len(self.genes):
            return self.genes.pop(index)
        raise IndexError("Gene index out of range")
    
    def get_active_genes(self) -> List[Gene]:
        """Get all active genes."""
        return [gene for gene in self.genes if gene.active]
    
    def to_atomspace_structure(self) -> List[Atom]:
        """Convert this individual to AtomSpace atoms."""
        return [gene.atom for gene in self.get_active_genes()]
    
    def copy(self) -> 'Individual':
        """Create a copy of this individual."""
        new_individual = Individual([gene.copy() for gene in self.genes])
        new_individual.fitness = self.fitness
        new_individual.age = self.age
        new_individual.generation = self.generation
        new_individual.metadata = copy.deepcopy(self.metadata)
        return new_individual
    
    def __len__(self) -> int:
        """Return the number of genes."""
        return len(self.genes)


class MutationOperator(EvolutionaryOperator):
    """Mutation operator for evolving individuals."""
    
    def __init__(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Initialize mutation operator.
        
        Args:
            mutation_rate: Probability of mutation occurring
            mutation_strength: Strength of mutations when they occur
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
    
    def apply(self, individual: Individual, context: Dict[str, Any] = None) -> Individual:
        """Apply mutation to an individual."""
        mutated = individual.copy()
        
        for gene in mutated.genes:
            if random.random() < self.mutation_rate:
                # Mutate gene weight
                gene.weight += random.gauss(0, self.mutation_strength)
                gene.weight = max(0.0, min(1.0, gene.weight))
                
                # Possibly toggle gene activity
                if random.random() < self.mutation_rate * 0.5:
                    gene.active = not gene.active
        
        return mutated
    
    def get_probability(self) -> float:
        """Get mutation probability."""
        return self.mutation_rate


class CrossoverOperator(EvolutionaryOperator):
    """Crossover operator for combining individuals."""
    
    def __init__(self, crossover_rate: float = 0.7):
        """Initialize crossover operator.
        
        Args:
            crossover_rate: Probability of crossover occurring
        """
        self.crossover_rate = crossover_rate
    
    def apply(self, individual: Individual, context: Dict[str, Any] = None) -> Individual:
        """Apply crossover between two individuals."""
        if context is None or 'partner' not in context:
            return individual.copy()
        
        partner = context['partner']
        
        # Create offspring through genetic crossover
        offspring = Individual()
        
        # Determine crossover points
        min_length = min(len(individual.genes), len(partner.genes))
        if min_length > 1:
            crossover_point = random.randint(1, min_length - 1)
            
            # Combine genes from both parents
            offspring.genes = (individual.genes[:crossover_point] + 
                             partner.genes[crossover_point:])
        else:
            # If one parent is empty or very short, copy from the other
            offspring.genes = individual.genes.copy() if individual.genes else partner.genes.copy()
        
        return offspring
    
    def get_probability(self) -> float:
        """Get crossover probability."""
        return self.crossover_rate


class SemanticAwarenessFitness(FitnessFunction):
    """Fitness function that considers semantic coherence of cognitive structures."""
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize with reference to AtomSpace for semantic evaluation.
        
        Args:
            atomspace: The AtomSpace for semantic context
        """
        self.atomspace = atomspace
    
    def evaluate(self, individual: Individual, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness based on semantic coherence and functionality."""
        if not individual.genes:
            return 0.0
        
        fitness = 0.0
        active_genes = individual.get_active_genes()
        
        if not active_genes:
            return 0.0
        
        # Base fitness from gene weights and activity
        base_fitness = sum(gene.weight for gene in active_genes) / len(active_genes)
        
        # Semantic coherence evaluation
        semantic_score = self._evaluate_semantic_coherence(active_genes)
        
        # Diversity bonus (reward for having diverse gene types)
        diversity_score = self._evaluate_diversity(active_genes)
        
        # Combine fitness components
        fitness = (0.4 * base_fitness + 
                  0.4 * semantic_score + 
                  0.2 * diversity_score)
        
        return max(0.0, min(1.0, fitness))
    
    def _evaluate_semantic_coherence(self, genes: List[Gene]) -> float:
        """Evaluate how semantically coherent the genes are together."""
        if len(genes) <= 1:
            return 1.0
        
        coherence = 0.0
        comparisons = 0
        
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene1, gene2 = genes[i], genes[j]
                
                # Check for semantic relationships in AtomSpace
                similarity = self._compute_atom_similarity(gene1.atom, gene2.atom)
                coherence += similarity
                comparisons += 1
        
        return coherence / comparisons if comparisons > 0 else 0.0
    
    def _evaluate_diversity(self, genes: List[Gene]) -> float:
        """Evaluate diversity of gene types."""
        if not genes:
            return 0.0
        
        atom_types = set(gene.atom.atom_type for gene in genes)
        return len(atom_types) / len(genes)
    
    def _compute_atom_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Compute similarity between two atoms."""
        # Simple similarity based on type and structure
        if atom1.atom_type == atom2.atom_type:
            if atom1.is_node() and atom2.is_node():
                return 0.8 if atom1.name == atom2.name else 0.5
            elif atom1.is_link() and atom2.is_link():
                # Compare link structures
                if len(atom1.outgoing_set) == len(atom2.outgoing_set):
                    return 0.7
                else:
                    return 0.3
            return 0.5
        return 0.2


class MosesEngine:
    """Main MOSES evolution engine for cognitive structure evolution."""
    
    def __init__(self, atomspace: AtomSpace, population_size: int = 50):
        """Initialize MOSES engine.
        
        Args:
            atomspace: The AtomSpace for cognitive structure management
            population_size: Size of evolutionary population
        """
        self.atomspace = atomspace
        self.population_size = population_size
        self.generation = 0
        
        # Evolutionary operators
        self.operators: List[EvolutionaryOperator] = [
            MutationOperator(mutation_rate=0.1),
            CrossoverOperator(crossover_rate=0.7)
        ]
        
        # Fitness function
        self.fitness_function = SemanticAwarenessFitness(atomspace)
        
        # Current population
        self.population: List[Individual] = []
        
        # Evolution parameters
        self.elite_size = max(1, population_size // 10)  # Keep top 10%
        self.selection_pressure = 1.5
        
    def initialize_population(self, seed_atoms: List[Atom] = None) -> None:
        """Initialize the evolution population.
        
        Args:
            seed_atoms: Optional seed atoms to start evolution from
        """
        self.population = []
        
        # Create initial population
        for _ in range(self.population_size):
            individual = Individual()
            
            # Add random genes from AtomSpace or seed atoms
            if seed_atoms:
                # Use seed atoms with some variation
                num_genes = random.randint(1, min(5, len(seed_atoms)))
                selected_atoms = random.sample(seed_atoms, num_genes)
                for atom in selected_atoms:
                    gene = Gene(
                        atom=atom,
                        weight=random.random(),
                        active=random.random() > 0.2  # 80% chance of being active
                    )
                    individual.add_gene(gene)
            else:
                # Create random genes from existing AtomSpace atoms
                all_atoms = self.atomspace.get_all_atoms()
                if all_atoms:
                    num_genes = random.randint(1, min(5, len(all_atoms)))
                    for _ in range(num_genes):
                        atom = random.choice(all_atoms)
                        gene = Gene(
                            atom=atom,
                            weight=random.random(),
                            active=True
                        )
                        individual.add_gene(gene)
                else:
                    # Create a basic gene if AtomSpace is empty
                    concept_node = Node("ConceptNode", f"concept_{random.randint(1, 1000)}")
                    gene = Gene(atom=concept_node, weight=random.random())
                    individual.add_gene(gene)
            
            individual.generation = self.generation
            self.population.append(individual)
    
    def evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in population."""
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.fitness_function.evaluate(individual)
    
    def select_parents(self) -> List[Individual]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = 3
        
        for _ in range(len(self.population)):
            # Tournament selection
            tournament = random.sample(self.population, 
                                     min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda ind: ind.fitness or 0.0)
            parents.append(winner)
        
        return parents
    
    def evolve_generation(self) -> None:
        """Evolve one generation of the population."""
        # Evaluate current population
        self.evaluate_population()
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)
        
        # Keep elite individuals
        new_population = self.population[:self.elite_size].copy()
        
        # Generate offspring to fill remaining population
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Apply crossover
            crossover_op = CrossoverOperator()
            if random.random() < crossover_op.get_probability():
                offspring = crossover_op.apply(parent1, {'partner': parent2})
            else:
                offspring = parent1.copy()
            
            # Apply mutation
            mutation_op = MutationOperator()
            if random.random() < mutation_op.get_probability():
                offspring = mutation_op.apply(offspring)
            
            # Update offspring metadata
            offspring.generation = self.generation + 1
            offspring.fitness = None  # Will be evaluated next generation
            
            new_population.append(offspring)
        
        # Update population and generation counter
        self.population = new_population
        self.generation += 1
    
    def _tournament_select(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, 
                                 min(tournament_size, len(self.population)))
        return max(tournament, key=lambda ind: ind.fitness or 0.0)
    
    def get_best_individual(self) -> Optional[Individual]:
        """Get the best individual from current population."""
        if not self.population:
            return None
        
        self.evaluate_population()
        return max(self.population, key=lambda ind: ind.fitness or 0.0)
    
    def get_population_stats(self) -> Dict[str, float]:
        """Get statistics about the current population."""
        if not self.population:
            return {}
        
        self.evaluate_population()
        fitnesses = [ind.fitness or 0.0 for ind in self.population]
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(fitnesses),
            'average_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'diversity': self._calculate_diversity()
        }
    
    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity in the population."""
        if len(self.population) <= 1:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                similarity = self._individual_similarity(
                    self.population[i], 
                    self.population[j]
                )
                total_similarity += similarity
                comparisons += 1
        
        average_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
        return 1.0 - average_similarity  # Diversity is inverse of similarity
    
    def _individual_similarity(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate similarity between two individuals."""
        genes1 = ind1.get_active_genes()
        genes2 = ind2.get_active_genes()
        
        if not genes1 and not genes2:
            return 1.0
        if not genes1 or not genes2:
            return 0.0
        
        # Simple Jaccard similarity based on atom types
        types1 = set(gene.atom.atom_type for gene in genes1)
        types2 = set(gene.atom.atom_type for gene in genes2)
        
        intersection = len(types1.intersection(types2))
        union = len(types1.union(types2))
        
        return intersection / union if union > 0 else 0.0


class EvolutionHistory:
    """Tracks the history of evolution for analysis and learning."""
    
    def __init__(self):
        """Initialize evolution history tracker."""
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_individuals: List[Individual] = []
        self.evolutionary_events: List[Dict[str, Any]] = []
    
    def record_generation(self, engine: MosesEngine) -> None:
        """Record statistics for a generation."""
        stats = engine.get_population_stats()
        self.generation_stats.append(stats)
        
        best_individual = engine.get_best_individual()
        if best_individual:
            self.best_individuals.append(best_individual.copy())
    
    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an evolutionary event."""
        event = {
            'type': event_type,
            'generation': data.get('generation', 0),
            'timestamp': data.get('timestamp'),
            'data': data
        }
        self.evolutionary_events.append(event)
    
    def get_fitness_trajectory(self) -> List[float]:
        """Get the trajectory of best fitness over generations."""
        return [stats.get('best_fitness', 0.0) for stats in self.generation_stats]
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns in evolution."""
        if len(self.generation_stats) < 10:
            return {'status': 'insufficient_data'}
        
        # Check for fitness plateaus
        recent_fitnesses = self.get_fitness_trajectory()[-10:]
        fitness_variance = sum((f - sum(recent_fitnesses)/10)**2 for f in recent_fitnesses) / 10
        
        # Check for diversity trends
        recent_diversity = [stats.get('diversity', 0.0) for stats in self.generation_stats[-10:]]
        diversity_trend = (recent_diversity[-1] - recent_diversity[0]) / len(recent_diversity)
        
        return {
            'status': 'analyzed',
            'fitness_variance': fitness_variance,
            'diversity_trend': diversity_trend,
            'generations_analyzed': len(self.generation_stats),
            'converged': fitness_variance < 0.001 and abs(diversity_trend) < 0.01
        }