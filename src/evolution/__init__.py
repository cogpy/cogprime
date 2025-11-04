"""
CogPrime Evolution Module - OpenCog MOSES Evolution Engine

This module implements an evolutionary computation engine inspired by OpenCog's MOSES
(Meta-Optimizing Semantic Evolutionary Search) system for evolving cognitive structures,
patterns, and behaviors.

The evolution module provides:
- Evolutionary algorithms for optimizing cognitive structures
- Fitness evaluation for cognitive patterns
- Population management and selection
- Integration with AtomSpace for evolutionary learning

Usage:
    from cogprime.evolution import MosesEngine, EvolutionaryOrchestrator
    
    # Create evolution engine
    moses = MosesEngine()
    
    # Create orchestrator for complex evolution
    orchestrator = EvolutionaryOrchestrator(moses)
"""

from .moses_engine import MosesEngine, EvolutionaryOperator, FitnessFunction
from .orchestrator import EvolutionaryOrchestrator, EvolutionConfig
from .population import Population, Individual, Gene
from .selection import SelectionStrategy, TournamentSelection, RouletteSelection

__all__ = [
    'MosesEngine',
    'EvolutionaryOperator',
    'FitnessFunction',
    'EvolutionaryOrchestrator',
    'EvolutionConfig',
    'Population',
    'Individual', 
    'Gene',
    'SelectionStrategy',
    'TournamentSelection',
    'RouletteSelection'
]