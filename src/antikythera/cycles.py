"""
Civilizational Cycles Implementation

This module implements the civilizational transformation cycles as described in
the Cycle of the Phoenix framework, modeling the historical patterns of rise,
peak, and fall phases across 600-year cycles.
"""

import time
import math
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from src.atomspace import AtomSpace, Node, Link, TruthValue, AttentionValue
from src.evolution import MosesEngine, Individual


class PhaseType(Enum):
    """Types of phases in civilizational cycles."""
    RISE = "rise"
    PEAK = "peak" 
    FALL = "fall"
    TRANSITION = "transition"


class CyclePattern(Enum):
    """Patterns observed across civilizational cycles."""
    INSTITUTIONAL_EMERGENCE = "institutional_emergence"
    CULTURAL_INTEGRATION = "cultural_integration"
    PARADIGM_BREAKDOWN = "paradigm_breakdown"
    INNOVATION_SYNTHESIS = "innovation_synthesis"
    MEANING_TRANSFORMATION = "meaning_transformation"


@dataclass
class PhaseCharacteristics:
    """Characteristics of a cycle phase."""
    duration_years: int
    energy_level: float  # 0.0 to 1.0
    stability: float     # 0.0 to 1.0  
    innovation_rate: float  # 0.0 to 1.0
    integration_level: float  # 0.0 to 1.0
    transformation_intensity: float  # 0.0 to 1.0
    dominant_patterns: List[CyclePattern] = field(default_factory=list)


class CyclePhase:
    """Represents a single phase within a civilizational cycle."""
    
    def __init__(self, phase_type: PhaseType, 
                 characteristics: PhaseCharacteristics,
                 name: str = ""):
        """Initialize a cycle phase.
        
        Args:
            phase_type: Type of this phase
            characteristics: Characteristics defining this phase
            name: Optional name for this phase
        """
        self.phase_type = phase_type
        self.characteristics = characteristics
        self.name = name or f"{phase_type.value}_phase"
        
        # Phase state
        self.start_time: Optional[float] = None
        self.current_progress = 0.0  # 0.0 to 1.0
        self.is_active = False
        
        # Cognitive structures active in this phase
        self.active_structures: List[Node] = []
        self.phase_patterns: Dict[str, Any] = {}
        
        # Evolutionary pressures
        self.selection_pressures: Dict[str, float] = {}
        self.mutation_rates: Dict[str, float] = {}
        
    def activate(self) -> None:
        """Activate this phase."""
        self.is_active = True
        self.start_time = time.time()
        self.current_progress = 0.0
    
    def deactivate(self) -> None:
        """Deactivate this phase."""
        self.is_active = False
        self.current_progress = 1.0
    
    def update_progress(self, elapsed_time: float) -> None:
        """Update phase progress based on elapsed time.
        
        Args:
            elapsed_time: Time elapsed since phase start (in simulation units)
        """
        if not self.is_active or self.start_time is None:
            return
        
        # Calculate progress (normalized time)
        phase_duration = self.characteristics.duration_years
        self.current_progress = min(1.0, elapsed_time / phase_duration)
    
    def get_current_energy(self) -> float:
        """Get current energy level based on phase progress."""
        base_energy = self.characteristics.energy_level
        
        if self.phase_type == PhaseType.RISE:
            # Energy increases during rise phase
            return base_energy * (0.3 + 0.7 * self.current_progress)
        elif self.phase_type == PhaseType.PEAK:
            # Energy remains high during peak
            return base_energy * (0.8 + 0.2 * math.sin(math.pi * self.current_progress))
        elif self.phase_type == PhaseType.FALL:
            # Energy decreases during fall phase
            return base_energy * (1.0 - 0.7 * self.current_progress)
        else:
            return base_energy
    
    def get_current_stability(self) -> float:
        """Get current stability level based on phase progress.""" 
        base_stability = self.characteristics.stability
        
        if self.phase_type == PhaseType.RISE:
            # Stability increases during rise
            return base_stability * (0.4 + 0.6 * self.current_progress)
        elif self.phase_type == PhaseType.PEAK:
            # High stability during peak
            return base_stability * (0.9 + 0.1 * math.cos(math.pi * self.current_progress))
        elif self.phase_type == PhaseType.FALL:
            # Stability decreases during fall
            return base_stability * (1.0 - 0.8 * self.current_progress)
        else:
            return base_stability
    
    def get_evolutionary_parameters(self) -> Dict[str, float]:
        """Get evolutionary parameters for current phase state."""
        energy = self.get_current_energy()
        stability = self.get_current_stability()
        
        return {
            'mutation_rate': (1.0 - stability) * 0.3 + 0.05,  # Higher when less stable
            'selection_pressure': stability * 1.5 + 0.5,      # Higher when more stable
            'crossover_rate': energy * 0.5 + 0.3,            # Higher when more energetic
            'population_growth': energy * stability * 0.2 + 0.8  # Growth factor
        }


class CivilizationalCycle:
    """Represents a complete civilizational cycle with rise/peak/fall phases."""
    
    def __init__(self, name: str, cycle_id: int = 0):
        """Initialize a civilizational cycle.
        
        Args:
            name: Name of this cycle
            cycle_id: Unique identifier for this cycle
        """
        self.name = name
        self.cycle_id = cycle_id
        self.total_duration = 600  # Standard 600-year cycle
        
        # Create the three standard phases
        self.phases = self._create_standard_phases()
        self.current_phase_index = 0
        
        # Cycle state
        self.start_time: Optional[float] = None
        self.is_active = False
        self.completed = False
        
        # Historical data from the Cycle of the Phoenix
        self.historical_periods = self._load_historical_periods()
        
        # Integration points with other systems
        self.cognitive_bindings: Dict[str, Any] = {}
        self.evolutionary_bindings: Dict[str, Any] = {}
        
    def _create_standard_phases(self) -> List[CyclePhase]:
        """Create the standard three-phase cycle structure."""
        phases = []
        
        # Rise Phase (200 years)
        rise_characteristics = PhaseCharacteristics(
            duration_years=200,
            energy_level=0.7,
            stability=0.5,
            innovation_rate=0.8,
            integration_level=0.6,
            transformation_intensity=0.7,
            dominant_patterns=[
                CyclePattern.INSTITUTIONAL_EMERGENCE,
                CyclePattern.INNOVATION_SYNTHESIS
            ]
        )
        rise_phase = CyclePhase(PhaseType.RISE, rise_characteristics, f"{self.name}_rise")
        phases.append(rise_phase)
        
        # Peak Phase (200 years)
        peak_characteristics = PhaseCharacteristics(
            duration_years=200,
            energy_level=0.9,
            stability=0.9,
            innovation_rate=0.5,
            integration_level=0.9,
            transformation_intensity=0.3,
            dominant_patterns=[
                CyclePattern.CULTURAL_INTEGRATION,
                CyclePattern.INSTITUTIONAL_EMERGENCE
            ]
        )
        peak_phase = CyclePhase(PhaseType.PEAK, peak_characteristics, f"{self.name}_peak")
        phases.append(peak_phase)
        
        # Fall Phase (200 years)  
        fall_characteristics = PhaseCharacteristics(
            duration_years=200,
            energy_level=0.4,
            stability=0.3,
            innovation_rate=0.9,
            integration_level=0.4,
            transformation_intensity=0.9,
            dominant_patterns=[
                CyclePattern.PARADIGM_BREAKDOWN,
                CyclePattern.MEANING_TRANSFORMATION
            ]
        )
        fall_phase = CyclePhase(PhaseType.FALL, fall_characteristics, f"{self.name}_fall")
        phases.append(fall_phase)
        
        return phases
    
    def _load_historical_periods(self) -> Dict[str, Dict[str, Any]]:
        """Load historical period data from Cycle of the Phoenix."""
        return {
            'cycle_1': {
                'period': (-1200, -600),
                'description': 'Bronze Age Collapse to Axial Age',
                'phases': {
                    'rise': (-1200, -1000, 'Bronze Age Collapse and Recovery'),
                    'peak': (-1000, -800, 'Early Iron Age Developments'),
                    'fall': (-800, -600, 'Emergence of Axial Thought')
                }
            },
            'cycle_2': {
                'period': (-600, 0),
                'description': 'Classical Period to Roman Empire',
                'phases': {
                    'rise': (-600, -400, 'Rise of Greek City-States'),
                    'peak': (-400, -200, 'Hellenistic Synthesis'),
                    'fall': (-200, 0, 'Roman Republican Period')
                }
            },
            'cycle_3': {
                'period': (0, 600),
                'description': 'Roman to Medieval',
                'phases': {
                    'rise': (0, 200, 'Early Roman Empire'),
                    'peak': (200, 400, 'Imperial Crisis and Recovery'),
                    'fall': (400, 600, 'Western Roman Collapse')
                }
            },
            'cycle_4': {
                'period': (600, 1200),
                'description': 'Medieval Period',
                'phases': {
                    'rise': (600, 800, 'Early Medieval Formation'),
                    'peak': (800, 1000, 'Carolingian Renaissance'),
                    'fall': (1000, 1200, 'First Medieval Transformation')
                }
            },
            'cycle_5': {
                'period': (1200, 1800),
                'description': 'Late Medieval to Modern',
                'phases': {
                    'rise': (1200, 1400, 'High Medieval Synthesis'),
                    'peak': (1400, 1600, 'Renaissance and Reformation'),
                    'fall': (1600, 1800, 'Scientific Revolution')
                }
            },
            'cycle_6': {
                'period': (1800, 2400),
                'description': 'Modern to Future',
                'phases': {
                    'rise': (1800, 2000, 'Industrial Revolution'),
                    'peak': (2000, 2200, 'Information Age'),
                    'fall': (2200, 2400, 'Future Integration')
                }
            }
        }
    
    def start_cycle(self) -> None:
        """Start the civilizational cycle."""
        self.is_active = True
        self.start_time = time.time()
        self.current_phase_index = 0
        self.completed = False
        
        # Activate first phase
        if self.phases:
            self.phases[0].activate()
    
    def stop_cycle(self) -> None:
        """Stop the civilizational cycle."""
        self.is_active = False
        
        # Deactivate current phase
        current_phase = self.get_current_phase()
        if current_phase:
            current_phase.deactivate()
    
    def update(self, elapsed_time: float) -> None:
        """Update cycle state based on elapsed time.
        
        Args:
            elapsed_time: Time elapsed since cycle start
        """
        if not self.is_active:
            return
        
        current_phase = self.get_current_phase()
        if not current_phase:
            return
        
        # Update current phase
        phase_start_time = sum(phase.characteristics.duration_years 
                              for phase in self.phases[:self.current_phase_index])
        phase_elapsed = elapsed_time - phase_start_time
        
        current_phase.update_progress(phase_elapsed)
        
        # Check for phase transition
        if current_phase.current_progress >= 1.0:
            self._transition_to_next_phase()
    
    def _transition_to_next_phase(self) -> None:
        """Transition to the next phase in the cycle."""
        # Deactivate current phase
        current_phase = self.get_current_phase()
        if current_phase:
            current_phase.deactivate()
        
        # Move to next phase
        self.current_phase_index += 1
        
        if self.current_phase_index >= len(self.phases):
            # Cycle completed
            self.completed = True
            self.is_active = False
        else:
            # Activate next phase
            next_phase = self.phases[self.current_phase_index]
            next_phase.activate()
    
    def get_current_phase(self) -> Optional[CyclePhase]:
        """Get the currently active phase."""
        if 0 <= self.current_phase_index < len(self.phases):
            return self.phases[self.current_phase_index]
        return None
    
    def get_cycle_progress(self) -> float:
        """Get overall progress through the cycle (0.0 to 1.0)."""
        if not self.phases:
            return 0.0
        
        completed_phases = self.current_phase_index
        current_phase = self.get_current_phase()
        
        if current_phase:
            phase_progress = current_phase.current_progress
        else:
            phase_progress = 1.0 if self.completed else 0.0
        
        total_progress = (completed_phases + phase_progress) / len(self.phases)
        return min(1.0, total_progress)
    
    def get_current_characteristics(self) -> Dict[str, float]:
        """Get current civilizational characteristics."""
        current_phase = self.get_current_phase()
        
        if not current_phase:
            return {
                'energy': 0.0,
                'stability': 0.0,
                'innovation_rate': 0.0,
                'integration_level': 0.0,
                'transformation_intensity': 0.0
            }
        
        return {
            'energy': current_phase.get_current_energy(),
            'stability': current_phase.get_current_stability(),
            'innovation_rate': current_phase.characteristics.innovation_rate,
            'integration_level': current_phase.characteristics.integration_level,
            'transformation_intensity': current_phase.characteristics.transformation_intensity
        }
    
    def bind_to_cognitive_system(self, atomspace: AtomSpace) -> None:
        """Bind cycle to cognitive system through AtomSpace.
        
        Args:
            atomspace: The AtomSpace for cognitive integration
        """
        # Create cycle representation in AtomSpace
        cycle_node = Node("ConceptNode", f"civilizational_cycle_{self.name}")
        atomspace.add(cycle_node)
        
        # Add phases to AtomSpace
        for i, phase in enumerate(self.phases):
            phase_node = Node("ConceptNode", f"cycle_phase_{phase.name}")
            
            # Link phase to cycle
            phase_link = Link("MemberLink", [phase_node, cycle_node])
            
            # Set truth values based on phase characteristics
            energy = phase.characteristics.energy_level
            stability = phase.characteristics.stability
            phase_node.set_truth_value(TruthValue(strength=energy, confidence=stability))
            
            atomspace.add(phase_node)
            atomspace.add(phase_link)
        
        self.cognitive_bindings['atomspace'] = atomspace
        self.cognitive_bindings['cycle_node'] = cycle_node
    
    def bind_to_evolution(self, moses_engine: MosesEngine) -> None:
        """Bind cycle to evolutionary system.
        
        Args:
            moses_engine: The MOSES evolution engine
        """
        self.evolutionary_bindings['moses_engine'] = moses_engine
        
        # Set up evolutionary parameter adaptation based on cycle phase
        def update_evolution_parameters():
            current_phase = self.get_current_phase()
            if current_phase:
                params = current_phase.get_evolutionary_parameters()
                
                # Update MOSES engine parameters
                for operator in moses_engine.operators:
                    if hasattr(operator, 'mutation_rate'):
                        operator.mutation_rate = params['mutation_rate']
                    if hasattr(operator, 'crossover_rate'):
                        operator.crossover_rate = params.get('crossover_rate', 0.7)
                
                moses_engine.selection_pressure = params['selection_pressure']
        
        self.evolutionary_bindings['parameter_updater'] = update_evolution_parameters
    
    def apply_evolutionary_pressure(self) -> None:
        """Apply evolutionary pressure based on current cycle phase."""
        if 'parameter_updater' in self.evolutionary_bindings:
            self.evolutionary_bindings['parameter_updater']()
    
    def get_historical_context(self, target_year: int) -> Optional[Dict[str, Any]]:
        """Get historical context for a specific year.
        
        Args:
            target_year: Year to get historical context for
            
        Returns:
            Historical context information or None
        """
        for cycle_name, cycle_data in self.historical_periods.items():
            start_year, end_year = cycle_data['period']
            
            if start_year <= target_year <= end_year:
                # Find appropriate phase
                phases = cycle_data['phases']
                for phase_name, phase_info in phases.items():
                    if len(phase_info) == 3:  # (start, end, description)
                        phase_start, phase_end, description = phase_info
                        if phase_start <= target_year <= phase_end:
                            return {
                                'cycle': cycle_name,
                                'cycle_description': cycle_data['description'],
                                'phase': phase_name,
                                'phase_description': description,
                                'year': target_year,
                                'phase_progress': (target_year - phase_start) / (phase_end - phase_start)
                            }
        
        return None


class MetaCycle:
    """Manages multiple civilizational cycles and their interactions."""
    
    def __init__(self, name: str = "meta_civilization"):
        """Initialize meta-cycle coordinator.
        
        Args:
            name: Name of the meta-cycle system
        """
        self.name = name
        self.cycles: List[CivilizationalCycle] = []
        self.cycle_interactions: Dict[Tuple[int, int], float] = {}
        
        # Meta-cycle state
        self.start_time: Optional[float] = None
        self.is_active = False
        
    def add_cycle(self, cycle: CivilizationalCycle) -> None:
        """Add a civilizational cycle to the meta-cycle.
        
        Args:
            cycle: The cycle to add
        """
        cycle.cycle_id = len(self.cycles)
        self.cycles.append(cycle)
    
    def set_cycle_interaction(self, cycle1_id: int, cycle2_id: int, 
                            interaction_strength: float) -> None:
        """Set interaction strength between two cycles.
        
        Args:
            cycle1_id: ID of first cycle
            cycle2_id: ID of second cycle
            interaction_strength: Strength of interaction (-1.0 to 1.0)
        """
        key = (min(cycle1_id, cycle2_id), max(cycle1_id, cycle2_id))
        self.cycle_interactions[key] = interaction_strength
    
    def start_meta_cycle(self) -> None:
        """Start all cycles in the meta-cycle."""
        self.is_active = True
        self.start_time = time.time()
        
        for cycle in self.cycles:
            cycle.start_cycle()
    
    def update_all_cycles(self, elapsed_time: float) -> None:
        """Update all cycles in the meta-cycle.
        
        Args:
            elapsed_time: Time elapsed since start
        """
        if not self.is_active:
            return
        
        # Update individual cycles
        for cycle in self.cycles:
            cycle.update(elapsed_time)
        
        # Apply inter-cycle interactions
        self._apply_cycle_interactions()
    
    def _apply_cycle_interactions(self) -> None:
        """Apply interactions between cycles."""
        for (cycle1_id, cycle2_id), strength in self.cycle_interactions.items():
            if (cycle1_id < len(self.cycles) and cycle2_id < len(self.cycles)):
                cycle1 = self.cycles[cycle1_id]
                cycle2 = self.cycles[cycle2_id]
                
                # Simple interaction: influence evolutionary parameters
                if cycle1.is_active and cycle2.is_active:
                    char1 = cycle1.get_current_characteristics()
                    char2 = cycle2.get_current_characteristics()
                    
                    # Cross-influence based on interaction strength
                    # This is a simplified model - could be much more sophisticated
                    pass
    
    def get_meta_characteristics(self) -> Dict[str, float]:
        """Get aggregate characteristics across all active cycles."""
        active_cycles = [cycle for cycle in self.cycles if cycle.is_active]
        
        if not active_cycles:
            return {
                'energy': 0.0,
                'stability': 0.0,
                'innovation_rate': 0.0,
                'integration_level': 0.0,
                'transformation_intensity': 0.0
            }
        
        # Aggregate characteristics
        total_chars = {
            'energy': 0.0,
            'stability': 0.0,
            'innovation_rate': 0.0,
            'integration_level': 0.0,
            'transformation_intensity': 0.0
        }
        
        for cycle in active_cycles:
            chars = cycle.get_current_characteristics()
            for key, value in chars.items():
                total_chars[key] += value
        
        # Average the characteristics
        count = len(active_cycles)
        return {key: value / count for key, value in total_chars.items()}