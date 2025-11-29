"""
Ontogenesis: Self-Generating Kernels

This module implements self-generating, evolving kernels through recursive
application of differential operators. Kernels can generate themselves,
optimize themselves, reproduce with other kernels, and evolve across generations.
"""

from .kernel import OntogeneticKernel, GeneratedKernel, DevelopmentStage
from .genome import (
    KernelGenome,
    KernelGene,
    CoefficientGene,
    OperatorGene,
    SymmetryGene,
    PreservationGene,
)
from .operators import (
    initialize_ontogenetic_kernel,
    self_generate,
    self_optimize,
    self_reproduce,
    ReproductionMethod,
)
from .evolution import (
    run_ontogenesis,
    OntogenesisConfig,
    EvolutionConfig,
    GenerationStats,
)

__all__ = [
    # Kernel classes
    "OntogeneticKernel",
    "GeneratedKernel",
    "DevelopmentStage",
    # Genome classes
    "KernelGenome",
    "KernelGene",
    "CoefficientGene",
    "OperatorGene",
    "SymmetryGene",
    "PreservationGene",
    # Operator functions
    "initialize_ontogenetic_kernel",
    "self_generate",
    "self_optimize",
    "self_reproduce",
    "ReproductionMethod",
    # Evolution
    "run_ontogenesis",
    "OntogenesisConfig",
    "EvolutionConfig",
    "GenerationStats",
]
