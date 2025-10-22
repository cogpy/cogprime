"""
Cycle of the Phoenix - Civilizational Transformation Framework

This module implements the complete Cycle of the Phoenix framework as an
antikythera orchestration system using the OpenCog MOSES evolution engine.

The Cycle of the Phoenix coordinates:
- 600-year civilizational cycles with rise/peak/fall phases
- MOSES evolutionary computation for cognitive structure evolution
- Antikythera orchestration for temporal coordination
- AtomSpace integration for knowledge representation

Usage:
    from cogprime.cycle_phoenix import CycleOfPhoenix, PhoenixOrchestrator
    
    # Create the main orchestrator
    phoenix = CycleOfPhoenix(atomspace)
    
    # Start the civilizational cycle orchestration
    phoenix.begin_cycle_of_transformation()
    
    # Monitor the evolution
    status = phoenix.get_transformation_status()
"""

from .phoenix_orchestrator import CycleOfPhoenix, PhoenixOrchestrator, TransformationMode
from .civilizational_engine import CivilizationalEngine, TransformationPhase
from .phoenix_moses import PhoenixMOSES, CivilizationalFitness

__all__ = [
    'CycleOfPhoenix',
    'PhoenixOrchestrator', 
    'TransformationMode',
    'CivilizationalEngine',
    'TransformationPhase',
    'PhoenixMOSES',
    'CivilizationalFitness'
]