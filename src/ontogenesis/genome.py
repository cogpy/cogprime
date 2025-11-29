"""
Genetic structures for ontogenetic kernels.

This module defines the genetic information (genome, genes) that enables
kernels to reproduce, mutate, and evolve.
"""

from dataclasses import dataclass, field
from typing import List, Any
from abc import ABC, abstractmethod
import numpy as np


class KernelGene(ABC):
    """Abstract base class for kernel genes."""

    @abstractmethod
    def mutate(self, mutation_rate: float) -> "KernelGene":
        """Create a mutated copy of this gene."""
        pass

    @abstractmethod
    def clone(self) -> "KernelGene":
        """Create an exact copy of this gene."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert gene to dictionary representation."""
        pass


@dataclass
class CoefficientGene(KernelGene):
    """Gene controlling a B-series coefficient (mutable)."""

    index: int
    value: float

    def mutate(self, mutation_rate: float) -> "CoefficientGene":
        """Mutate the coefficient value."""
        if np.random.random() < mutation_rate:
            # Apply ±10% mutation
            mutation = (np.random.random() - 0.5) * 0.2
            new_value = self.value * (1 + mutation)
            return CoefficientGene(index=self.index, value=new_value)
        return self.clone()

    def clone(self) -> "CoefficientGene":
        """Create exact copy."""
        return CoefficientGene(index=self.index, value=self.value)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"type": "coefficient", "index": self.index, "value": self.value}


@dataclass
class OperatorGene(KernelGene):
    """Gene controlling differential operators (mutable)."""

    operator_type: str  # 'chain', 'product', 'quotient'
    weight: float

    def mutate(self, mutation_rate: float) -> "OperatorGene":
        """Mutate the operator weight."""
        if np.random.random() < mutation_rate:
            mutation = (np.random.random() - 0.5) * 0.2
            new_weight = max(0.0, min(1.0, self.weight * (1 + mutation)))
            return OperatorGene(operator_type=self.operator_type, weight=new_weight)
        return self.clone()

    def clone(self) -> "OperatorGene":
        """Create exact copy."""
        return OperatorGene(operator_type=self.operator_type, weight=self.weight)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"type": "operator", "operator_type": self.operator_type, "weight": self.weight}


@dataclass
class SymmetryGene(KernelGene):
    """Gene preserving domain symmetries (immutable)."""

    symmetry_type: str
    properties: dict = field(default_factory=dict)

    def mutate(self, mutation_rate: float) -> "SymmetryGene":
        """Symmetry genes don't mutate."""
        return self.clone()

    def clone(self) -> "SymmetryGene":
        """Create exact copy."""
        return SymmetryGene(
            symmetry_type=self.symmetry_type, properties=self.properties.copy()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "symmetry",
            "symmetry_type": self.symmetry_type,
            "properties": self.properties,
        }


@dataclass
class PreservationGene(KernelGene):
    """Gene maintaining conserved quantities (immutable)."""

    quantity: str
    value: float

    def mutate(self, mutation_rate: float) -> "PreservationGene":
        """Preservation genes don't mutate."""
        return self.clone()

    def clone(self) -> "PreservationGene":
        """Create exact copy."""
        return PreservationGene(quantity=self.quantity, value=self.value)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"type": "preservation", "quantity": self.quantity, "value": self.value}


@dataclass
class KernelGenome:
    """The 'DNA' of a kernel containing all genetic information."""

    id: str  # Unique identifier
    generation: int  # Generation number
    lineage: List[str]  # Parent IDs
    genes: List[KernelGene]  # Genetic information
    fitness: float  # Overall fitness
    age: int  # Age in generations

    def get_coefficients(self) -> np.ndarray:
        """Extract coefficient values from coefficient genes."""
        coeff_genes = [g for g in self.genes if isinstance(g, CoefficientGene)]
        coeff_genes.sort(key=lambda g: g.index)
        return np.array([g.value for g in coeff_genes])

    def update_fitness(self, new_fitness: float) -> None:
        """Update the fitness value."""
        self.fitness = new_fitness

    def increment_age(self) -> None:
        """Increment the age by one generation."""
        self.age += 1

    def clone(self) -> "KernelGenome":
        """Create a deep copy of the genome."""
        return KernelGenome(
            id=f"{self.id}_clone",
            generation=self.generation,
            lineage=self.lineage.copy(),
            genes=[g.clone() for g in self.genes],
            fitness=self.fitness,
            age=self.age,
        )

    def to_dict(self) -> dict:
        """Convert genome to dictionary representation."""
        return {
            "id": self.id,
            "generation": self.generation,
            "lineage": self.lineage,
            "genes": [g.to_dict() for g in self.genes],
            "fitness": self.fitness,
            "age": self.age,
        }


def crossover_genomes(
    parent1: KernelGenome, parent2: KernelGenome, crossover_point: int = None
) -> tuple[KernelGenome, KernelGenome]:
    """Perform single-point crossover on two genomes.

    Args:
        parent1: First parent genome
        parent2: Second parent genome
        crossover_point: Point to split genes (random if None)

    Returns:
        Tuple of two offspring genomes
    """
    # Extract coefficient genes from both parents
    coeff1 = [g for g in parent1.genes if isinstance(g, CoefficientGene)]
    coeff2 = [g for g in parent2.genes if isinstance(g, CoefficientGene)]

    # Ensure same length
    min_len = min(len(coeff1), len(coeff2))
    coeff1 = coeff1[:min_len]
    coeff2 = coeff2[:min_len]

    # Choose crossover point
    if crossover_point is None:
        crossover_point = np.random.randint(1, min_len)

    # Create offspring genes
    offspring1_genes = coeff1[:crossover_point] + coeff2[crossover_point:]
    offspring2_genes = coeff2[:crossover_point] + coeff1[crossover_point:]

    # Add non-mutable genes from parent1
    for gene in parent1.genes:
        if not isinstance(gene, CoefficientGene):
            offspring1_genes.append(gene.clone())
            offspring2_genes.append(gene.clone())

    # Create offspring genomes
    offspring1 = KernelGenome(
        id=f"{parent1.id}_x_{parent2.id}_1",
        generation=max(parent1.generation, parent2.generation) + 1,
        lineage=[parent1.id, parent2.id],
        genes=offspring1_genes,
        fitness=0.0,  # Will be evaluated
        age=0,
    )

    offspring2 = KernelGenome(
        id=f"{parent1.id}_x_{parent2.id}_2",
        generation=max(parent1.generation, parent2.generation) + 1,
        lineage=[parent1.id, parent2.id],
        genes=offspring2_genes,
        fitness=0.0,  # Will be evaluated
        age=0,
    )

    return offspring1, offspring2


def mutate_genome(genome: KernelGenome, mutation_rate: float) -> KernelGenome:
    """Apply mutation to a genome.

    Args:
        genome: Genome to mutate
        mutation_rate: Probability of mutation per gene

    Returns:
        Mutated genome
    """
    mutated_genes = [gene.mutate(mutation_rate) for gene in genome.genes]

    return KernelGenome(
        id=f"{genome.id}_mut",
        generation=genome.generation,
        lineage=genome.lineage.copy(),
        genes=mutated_genes,
        fitness=0.0,  # Will be re-evaluated
        age=genome.age,
    )


def genetic_distance(genome1: KernelGenome, genome2: KernelGenome) -> float:
    """Calculate genetic distance between two genomes.

    Distance is measured as the Euclidean distance between coefficient values.

    Args:
        genome1: First genome
        genome2: Second genome

    Returns:
        Genetic distance (0.0 = identical, higher = more different)
    """
    coeffs1 = genome1.get_coefficients()
    coeffs2 = genome2.get_coefficients()

    # Handle different lengths
    min_len = min(len(coeffs1), len(coeffs2))
    if min_len == 0:
        return 0.0

    return np.linalg.norm(coeffs1[:min_len] - coeffs2[:min_len])
