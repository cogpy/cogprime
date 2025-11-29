"""Tests for genetic operations."""

import pytest
import numpy as np

from src.ontogenesis.genome import (
    KernelGenome,
    CoefficientGene,
    OperatorGene,
    SymmetryGene,
    PreservationGene,
    crossover_genomes,
    mutate_genome,
    genetic_distance,
)


def test_coefficient_gene_mutation():
    """Test coefficient gene mutation."""
    gene = CoefficientGene(index=0, value=1.0)

    # Mutate with 100% rate
    mutated = gene.mutate(mutation_rate=1.0)
    assert mutated.index == gene.index
    assert mutated.value != gene.value  # Should be different

    # Mutate with 0% rate
    not_mutated = gene.mutate(mutation_rate=0.0)
    assert not_mutated.value == gene.value


def test_coefficient_gene_clone():
    """Test coefficient gene cloning."""
    gene = CoefficientGene(index=0, value=1.5)
    clone = gene.clone()

    assert clone.index == gene.index
    assert clone.value == gene.value
    assert clone is not gene


def test_operator_gene_mutation():
    """Test operator gene mutation."""
    gene = OperatorGene(operator_type="chain", weight=0.5)

    mutated = gene.mutate(mutation_rate=1.0)
    assert mutated.operator_type == gene.operator_type
    # Weight should be different but still in [0, 1]
    assert 0.0 <= mutated.weight <= 1.0


def test_symmetry_gene_immutable():
    """Test that symmetry genes don't mutate."""
    gene = SymmetryGene(symmetry_type="rotation", properties={"angle": 90})

    mutated = gene.mutate(mutation_rate=1.0)
    assert mutated.symmetry_type == gene.symmetry_type
    assert mutated.properties == gene.properties


def test_preservation_gene_immutable():
    """Test that preservation genes don't mutate."""
    gene = PreservationGene(quantity="energy", value=1.0)

    mutated = gene.mutate(mutation_rate=1.0)
    assert mutated.quantity == gene.quantity
    assert mutated.value == gene.value


def test_kernel_genome_creation():
    """Test KernelGenome creation."""
    genes = [CoefficientGene(index=i, value=float(i)) for i in range(4)]

    genome = KernelGenome(
        id="test_genome", generation=0, lineage=[], genes=genes, fitness=0.8, age=0
    )

    assert genome.id == "test_genome"
    assert genome.generation == 0
    assert len(genome.genes) == 4
    assert genome.fitness == 0.8
    assert genome.age == 0


def test_kernel_genome_get_coefficients():
    """Test extracting coefficients from genome."""
    genes = [
        CoefficientGene(index=0, value=1.0),
        CoefficientGene(index=1, value=2.0),
        CoefficientGene(index=2, value=3.0),
    ]

    genome = KernelGenome(
        id="test", generation=0, lineage=[], genes=genes, fitness=0.8, age=0
    )

    coeffs = genome.get_coefficients()
    assert len(coeffs) == 3
    assert np.allclose(coeffs, [1.0, 2.0, 3.0])


def test_kernel_genome_update_fitness():
    """Test fitness update."""
    genes = [CoefficientGene(index=0, value=1.0)]
    genome = KernelGenome(
        id="test", generation=0, lineage=[], genes=genes, fitness=0.5, age=0
    )

    genome.update_fitness(0.9)
    assert genome.fitness == 0.9


def test_kernel_genome_increment_age():
    """Test age increment."""
    genes = [CoefficientGene(index=0, value=1.0)]
    genome = KernelGenome(
        id="test", generation=0, lineage=[], genes=genes, fitness=0.5, age=0
    )

    genome.increment_age()
    assert genome.age == 1

    genome.increment_age()
    assert genome.age == 2


def test_kernel_genome_clone():
    """Test genome cloning."""
    genes = [CoefficientGene(index=i, value=float(i)) for i in range(3)]
    genome = KernelGenome(
        id="original", generation=2, lineage=["parent1"], genes=genes, fitness=0.7, age=5
    )

    clone = genome.clone()

    assert clone.id == "original_clone"
    assert clone.generation == genome.generation
    assert clone.fitness == genome.fitness
    assert clone.age == genome.age
    assert len(clone.genes) == len(genome.genes)
    assert clone is not genome


def test_crossover_genomes():
    """Test genetic crossover."""
    genes1 = [CoefficientGene(index=i, value=1.0) for i in range(5)]
    genes2 = [CoefficientGene(index=i, value=2.0) for i in range(5)]

    genome1 = KernelGenome(
        id="parent1", generation=0, lineage=[], genes=genes1, fitness=0.8, age=1
    )
    genome2 = KernelGenome(
        id="parent2", generation=0, lineage=[], genes=genes2, fitness=0.7, age=1
    )

    offspring1, offspring2 = crossover_genomes(genome1, genome2, crossover_point=2)

    # Check generation increased
    assert offspring1.generation == 1
    assert offspring2.generation == 1

    # Check lineage
    assert len(offspring1.lineage) == 2
    assert "parent1" in offspring1.lineage
    assert "parent2" in offspring1.lineage

    # Check coefficients are mixed
    coeffs1 = offspring1.get_coefficients()
    coeffs2 = offspring2.get_coefficients()

    # First 2 from parent1, rest from parent2 for offspring1
    assert np.allclose(coeffs1[:2], [1.0, 1.0])
    assert np.allclose(coeffs1[2:], [2.0, 2.0, 2.0])

    # First 2 from parent2, rest from parent1 for offspring2
    assert np.allclose(coeffs2[:2], [2.0, 2.0])
    assert np.allclose(coeffs2[2:], [1.0, 1.0, 1.0])


def test_mutate_genome():
    """Test genome mutation."""
    genes = [CoefficientGene(index=i, value=1.0) for i in range(5)]
    genome = KernelGenome(
        id="original", generation=0, lineage=[], genes=genes, fitness=0.8, age=1
    )

    mutated = mutate_genome(genome, mutation_rate=1.0)

    assert mutated.id == "original_mut"
    assert mutated.generation == genome.generation
    assert len(mutated.genes) == len(genome.genes)

    # At least some genes should be mutated with high mutation rate
    original_coeffs = genome.get_coefficients()
    mutated_coeffs = mutated.get_coefficients()
    assert not np.allclose(original_coeffs, mutated_coeffs)


def test_genetic_distance():
    """Test genetic distance calculation."""
    genes1 = [CoefficientGene(index=i, value=1.0) for i in range(5)]
    genes2 = [CoefficientGene(index=i, value=1.0) for i in range(5)]

    genome1 = KernelGenome(
        id="g1", generation=0, lineage=[], genes=genes1, fitness=0.8, age=1
    )
    genome2 = KernelGenome(
        id="g2", generation=0, lineage=[], genes=genes2, fitness=0.8, age=1
    )

    # Identical genomes should have distance 0
    distance = genetic_distance(genome1, genome2)
    assert distance == 0.0

    # Change one gene
    genes2[0] = CoefficientGene(index=0, value=2.0)
    genome2.genes = genes2

    distance = genetic_distance(genome1, genome2)
    assert distance > 0.0


def test_gene_to_dict():
    """Test gene serialization."""
    coeff_gene = CoefficientGene(index=0, value=1.5)
    assert coeff_gene.to_dict() == {"type": "coefficient", "index": 0, "value": 1.5}

    op_gene = OperatorGene(operator_type="chain", weight=0.7)
    assert op_gene.to_dict() == {"type": "operator", "operator_type": "chain", "weight": 0.7}

    sym_gene = SymmetryGene(symmetry_type="rotation", properties={"angle": 90})
    assert sym_gene.to_dict() == {
        "type": "symmetry",
        "symmetry_type": "rotation",
        "properties": {"angle": 90},
    }

    pres_gene = PreservationGene(quantity="energy", value=1.0)
    assert pres_gene.to_dict() == {"type": "preservation", "quantity": "energy", "value": 1.0}


def test_genome_to_dict():
    """Test genome serialization."""
    genes = [CoefficientGene(index=i, value=float(i)) for i in range(3)]
    genome = KernelGenome(
        id="test", generation=2, lineage=["parent1"], genes=genes, fitness=0.8, age=5
    )

    genome_dict = genome.to_dict()

    assert genome_dict["id"] == "test"
    assert genome_dict["generation"] == 2
    assert genome_dict["lineage"] == ["parent1"]
    assert len(genome_dict["genes"]) == 3
    assert genome_dict["fitness"] == 0.8
    assert genome_dict["age"] == 5
