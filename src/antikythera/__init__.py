"""
Antikythera Orchestration Framework

This module implements an orchestration framework inspired by the Antikythera mechanism,
designed to coordinate complex civilizational transformation cycles as described in
the Cycle of the Phoenix framework.

The Antikythera framework provides:
- Cyclic coordination of cognitive processes
- Temporal pattern management across multiple scales
- Integration of evolutionary and civilizational dynamics
- Orchestration of rise/peak/fall phases in cognitive development

Usage:
    from cogprime.antikythera import AntikytheraOrchestrator, CivilizationalCycle
    
    # Create orchestrator for cycle management
    orchestrator = AntikytheraOrchestrator(atomspace)
    
    # Define and run civilizational cycles
    cycle = CivilizationalCycle("modern_cycle", phases=3)
    orchestrator.add_cycle(cycle)
    orchestrator.start_orchestration()
"""

from .orchestrator import AntikytheraOrchestrator, OrchestrationConfig
from .cycles import CivilizationalCycle, CyclePhase, PhaseType
from .temporal import TemporalCoordinator, TimeScale, TemporalPattern
from .integration import CognitiveIntegration, EvolutionaryIntegration

__all__ = [
    'AntikytheraOrchestrator',
    'OrchestrationConfig',
    'CivilizationalCycle', 
    'CyclePhase',
    'PhaseType',
    'TemporalCoordinator',
    'TimeScale',
    'TemporalPattern',
    'CognitiveIntegration',
    'EvolutionaryIntegration'
]