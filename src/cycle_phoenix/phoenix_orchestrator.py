"""
Cycle of the Phoenix - Main Orchestrator

This module implements the main orchestrator for the Cycle of the Phoenix framework,
integrating civilizational cycles, evolutionary computation, and cognitive architecture
into a unified transformation system.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.atomspace import AtomSpace
from src.evolution import EvolutionaryOrchestrator, EvolutionConfig
from src.antikythera import AntikytheraOrchestrator, OrchestrationConfig, CivilizationalCycle
from .civilizational_engine import CivilizationalEngine, TransformationPhase
from .phoenix_moses import PhoenixMOSES, CivilizationalFitness

# Configure logging
logger = logging.getLogger(__name__)


class TransformationMode(Enum):
    """Modes of civilizational transformation."""
    HISTORICAL_ANALYSIS = "historical_analysis"
    REAL_TIME_EVOLUTION = "real_time_evolution"
    PREDICTIVE_MODELING = "predictive_modeling"
    CONSCIOUSNESS_EVOLUTION = "consciousness_evolution"


@dataclass 
class PhoenixConfiguration:
    """Configuration for the Cycle of Phoenix orchestrator."""
    transformation_mode: TransformationMode = TransformationMode.CONSCIOUSNESS_EVOLUTION
    
    # Cycle parameters
    cycle_duration_years: int = 600
    phase_duration_years: int = 200
    enable_meta_cycles: bool = True
    
    # Evolution parameters
    population_size: int = 100
    max_generations: int = 500
    enable_civilizational_fitness: bool = True
    
    # Orchestration parameters
    time_acceleration: float = 1000.0  # Accelerate time for simulation
    enable_attention_economy: bool = True
    save_transformation_history: bool = True
    
    # Integration parameters
    sync_interval_seconds: float = 1.0
    auto_adapt_parameters: bool = True


class PhoenixOrchestrator:
    """
    Core orchestrator for the Cycle of the Phoenix framework.
    
    This orchestrator coordinates the entire transformation process, managing
    civilizational cycles, evolutionary computation, and cognitive evolution
    in an integrated framework.
    """
    
    def __init__(self, atomspace: AtomSpace, config: PhoenixConfiguration = None):
        """Initialize the Phoenix orchestrator.
        
        Args:
            atomspace: The AtomSpace for cognitive operations
            config: Configuration for the Phoenix system
        """
        self.atomspace = atomspace
        self.config = config or PhoenixConfiguration()
        
        # Core components
        self.civilizational_engine = CivilizationalEngine()
        self.phoenix_moses = PhoenixMOSES(atomspace)
        
        # Orchestration components
        self.antikythera_orchestrator: Optional[AntikytheraOrchestrator] = None
        self.evolutionary_orchestrator: Optional[EvolutionaryOrchestrator] = None
        
        # Phoenix state
        self.is_active = False
        self.start_time: Optional[float] = None
        self.current_cycle: Optional[CivilizationalCycle] = None
        self.transformation_history: List[Dict[str, Any]] = []
        
        # Callbacks for transformation events
        self.transformation_callbacks: List[Callable] = []
        self.cycle_completion_callbacks: List[Callable] = []
        
        # Initialize components
        self._initialize_orchestration_components()
    
    def _initialize_orchestration_components(self) -> None:
        """Initialize the orchestration components."""
        # Configure Antikythera orchestrator
        antikythera_config = OrchestrationConfig(
            time_acceleration=self.config.time_acceleration,
            enable_evolution=True,
            enable_cognitive_integration=True,
            max_cycles=6  # Standard 6 historical cycles
        )
        
        self.antikythera_orchestrator = AntikytheraOrchestrator(
            self.atomspace, 
            antikythera_config
        )
        
        # Configure evolutionary orchestrator
        evolution_config = EvolutionConfig(
            population_size=self.config.population_size,
            max_generations=self.config.max_generations,
            target_fitness=0.95,
            stagnation_limit=50,
            adaptive_parameters=True
        )
        
        self.evolutionary_orchestrator = EvolutionaryOrchestrator(
            self.atomspace,
            evolution_config
        )
        
        # Integrate evolution with Antikythera
        self.antikythera_orchestrator.integrate_evolution(self.evolutionary_orchestrator)
        
        # Setup Phoenix-specific integration
        self._setup_phoenix_integration()
        
        logger.info("Phoenix orchestration components initialized")
    
    def _setup_phoenix_integration(self) -> None:
        """Setup Phoenix-specific integration between components."""
        # Register callbacks for civilizational events
        def on_transformation_phase_change(phase_info: Dict[str, Any]) -> None:
            """Handle transformation phase changes."""
            phase = phase_info.get('phase')
            cycle_id = phase_info.get('cycle_id', 0)
            
            logger.info(f"Transformation phase changed to {phase} in cycle {cycle_id}")
            
            # Update Phoenix MOSES fitness function
            if self.config.enable_civilizational_fitness:
                self.phoenix_moses.update_civilizational_context(phase_info)
            
            # Trigger transformation callbacks
            for callback in self.transformation_callbacks:
                try:
                    callback(phase_info)
                except Exception as e:
                    logger.error(f"Error in transformation callback: {e}")
        
        # Register civilizational engine callback
        self.civilizational_engine.register_phase_callback(on_transformation_phase_change)
        
        # Register Antikythera callbacks
        if self.antikythera_orchestrator:
            def on_cycle_update(update_info: Dict[str, Any]) -> None:
                """Handle cycle updates from Antikythera."""
                # Update civilizational engine state
                self.civilizational_engine.sync_with_antikythera(update_info)
                
                # Record transformation history
                if self.config.save_transformation_history:
                    history_entry = {
                        'timestamp': time.time(),
                        'simulation_time': update_info.get('simulation_time', 0),
                        'transformation_state': self.get_transformation_status()
                    }
                    self.transformation_history.append(history_entry)
                    
                    # Limit history size
                    if len(self.transformation_history) > 10000:
                        self.transformation_history = self.transformation_history[-5000:]
            
            self.antikythera_orchestrator.register_cycle_callback(on_cycle_update)
    
    def begin_cycle_of_transformation(self) -> None:
        """Begin the Cycle of Phoenix transformation process."""
        if self.is_active:
            logger.warning("Cycle of Phoenix is already active")
            return
        
        logger.info("Beginning Cycle of Phoenix transformation")
        
        self.is_active = True
        self.start_time = time.time()
        
        # Initialize civilizational cycles based on mode
        self._initialize_civilizational_cycles()
        
        # Start evolutionary orchestration
        if self.evolutionary_orchestrator:
            # Add populations for different aspects of consciousness evolution
            self._setup_consciousness_evolution_populations()
            self.evolutionary_orchestrator.start_evolution(async_mode=True)
        
        # Start Antikythera orchestration
        if self.antikythera_orchestrator:
            self.antikythera_orchestrator.start_orchestration(async_mode=True)
        
        # Start civilizational engine
        self.civilizational_engine.start_transformation()
        
        logger.info("Cycle of Phoenix transformation has begun")
    
    def _initialize_civilizational_cycles(self) -> None:
        """Initialize civilizational cycles based on transformation mode."""
        if not self.antikythera_orchestrator:
            return
        
        if self.config.transformation_mode == TransformationMode.HISTORICAL_ANALYSIS:
            # Use historical cycles from the Cycle of the Phoenix document
            self._create_historical_cycles()
        
        elif self.config.transformation_mode == TransformationMode.CONSCIOUSNESS_EVOLUTION:
            # Create cycles focused on consciousness evolution
            self._create_consciousness_evolution_cycles()
        
        elif self.config.transformation_mode == TransformationMode.PREDICTIVE_MODELING:
            # Create predictive cycles for future modeling
            self._create_predictive_cycles()
        
        elif self.config.transformation_mode == TransformationMode.REAL_TIME_EVOLUTION:
            # Create adaptive cycles for real-time evolution
            self._create_adaptive_cycles()
    
    def _create_historical_cycles(self) -> None:
        """Create the six historical civilizational cycles."""
        historical_cycles = [
            ("Bronze Age to Axial Age", -1200, -600),
            ("Classical Period", -600, 0),
            ("Roman to Medieval", 0, 600),
            ("Medieval Period", 600, 1200),
            ("Late Medieval to Modern", 1200, 1800),
            ("Modern to Future", 1800, 2400)
        ]
        
        for i, (name, start_year, end_year) in enumerate(historical_cycles):
            cycle = CivilizationalCycle(name, cycle_id=i)
            
            # Set historical context
            cycle.metadata = {
                'historical_period': (start_year, end_year),
                'transformation_type': 'historical_analysis'
            }
            
            self.antikythera_orchestrator.add_cycle(cycle)
            
            if i == 0:  # First cycle becomes current
                self.current_cycle = cycle
    
    def _create_consciousness_evolution_cycles(self) -> None:
        """Create cycles focused on consciousness evolution."""
        consciousness_cycles = [
            ("Primitive Awareness", "Instinctual consciousness"),
            ("Symbolic Thinking", "Language and symbol emergence"),
            ("Rational Consciousness", "Logical and analytical thinking"),
            ("Integral Awareness", "Holistic and systems thinking"),
            ("Trans-rational Consciousness", "Beyond rational limitations"),
            ("Cosmic Consciousness", "Universal awareness integration")
        ]
        
        for i, (name, description) in enumerate(consciousness_cycles):
            cycle = CivilizationalCycle(f"Consciousness_{name.replace(' ', '_')}", cycle_id=i)
            
            # Configure for consciousness evolution
            cycle.metadata = {
                'consciousness_level': i + 1,
                'description': description,
                'transformation_type': 'consciousness_evolution'
            }
            
            # Adjust cycle parameters for consciousness evolution
            for phase in cycle.phases:
                # Higher consciousness levels have different characteristics
                consciousness_factor = (i + 1) / len(consciousness_cycles)
                
                phase.characteristics.integration_level *= (0.5 + 0.5 * consciousness_factor)
                phase.characteristics.transformation_intensity *= (1.5 - 0.5 * consciousness_factor)
            
            self.antikythera_orchestrator.add_cycle(cycle)
            
            if i == 2:  # Start at rational consciousness level
                self.current_cycle = cycle
    
    def _create_predictive_cycles(self) -> None:
        """Create predictive cycles for future modeling."""
        # Implementation for predictive modeling cycles
        future_cycle = CivilizationalCycle("Future_Transformation", cycle_id=0)
        future_cycle.metadata = {
            'transformation_type': 'predictive_modeling',
            'prediction_horizon': 200  # 200 years ahead
        }
        
        self.antikythera_orchestrator.add_cycle(future_cycle)
        self.current_cycle = future_cycle
    
    def _create_adaptive_cycles(self) -> None:
        """Create adaptive cycles for real-time evolution."""
        # Implementation for real-time adaptive cycles
        adaptive_cycle = CivilizationalCycle("Adaptive_Evolution", cycle_id=0)
        adaptive_cycle.metadata = {
            'transformation_type': 'real_time_evolution',
            'adaptation_rate': 'high'
        }
        
        self.antikythera_orchestrator.add_cycle(adaptive_cycle)
        self.current_cycle = adaptive_cycle
    
    def _setup_consciousness_evolution_populations(self) -> None:
        """Setup evolutionary populations for consciousness evolution."""
        if not self.evolutionary_orchestrator:
            return
        
        # Population for pattern recognition evolution
        self.evolutionary_orchestrator.add_population(
            size=self.config.population_size,
            initializer=None,  # Will use default
            selection_strategy=None
        )
        
        # Population for reasoning evolution  
        self.evolutionary_orchestrator.add_population(
            size=self.config.population_size // 2,
            initializer=None,
            selection_strategy=None
        )
        
        # Population for integration capabilities
        self.evolutionary_orchestrator.add_population(
            size=self.config.population_size // 2,
            initializer=None,
            selection_strategy=None
        )
    
    def pause_transformation(self) -> None:
        """Pause the transformation process."""
        if not self.is_active:
            return
        
        logger.info("Pausing Cycle of Phoenix transformation")
        
        # Pause components
        if self.antikythera_orchestrator:
            self.antikythera_orchestrator.stop_orchestration()
        
        if self.evolutionary_orchestrator:
            self.evolutionary_orchestrator.stop_evolution()
        
        self.civilizational_engine.pause_transformation()
        
        self.is_active = False
        logger.info("Cycle of Phoenix transformation paused")
    
    def resume_transformation(self) -> None:
        """Resume the transformation process."""
        if self.is_active:
            return
        
        logger.info("Resuming Cycle of Phoenix transformation")
        
        self.is_active = True
        
        # Resume components
        if self.antikythera_orchestrator:
            self.antikythera_orchestrator.start_orchestration(async_mode=True)
        
        if self.evolutionary_orchestrator:
            self.evolutionary_orchestrator.start_evolution(async_mode=True)
        
        self.civilizational_engine.resume_transformation()
        
        logger.info("Cycle of Phoenix transformation resumed")
    
    def complete_transformation(self) -> Dict[str, Any]:
        """Complete the transformation process and return results."""
        logger.info("Completing Cycle of Phoenix transformation")
        
        # Stop all components
        self.pause_transformation()
        
        # Generate completion report
        completion_report = {
            'completion_time': time.time(),
            'total_runtime': time.time() - (self.start_time or 0),
            'transformation_mode': self.config.transformation_mode.value,
            'final_status': self.get_transformation_status(),
            'cycles_completed': self._count_completed_cycles(),
            'evolutionary_results': self._get_evolutionary_results(),
            'civilizational_insights': self.civilizational_engine.get_transformation_insights()
        }
        
        # Trigger completion callbacks
        for callback in self.cycle_completion_callbacks:
            try:
                callback(completion_report)
            except Exception as e:
                logger.error(f"Error in cycle completion callback: {e}")
        
        logger.info("Cycle of Phoenix transformation completed")
        return completion_report
    
    def get_transformation_status(self) -> Dict[str, Any]:
        """Get current transformation status.
        
        Returns:
            Dictionary containing current transformation status
        """
        status = {
            'is_active': self.is_active,
            'transformation_mode': self.config.transformation_mode.value,
            'runtime': time.time() - (self.start_time or 0) if self.start_time else 0,
        }
        
        # Add current cycle information
        if self.current_cycle:
            status['current_cycle'] = {
                'name': self.current_cycle.name,
                'progress': self.current_cycle.get_cycle_progress(),
                'characteristics': self.current_cycle.get_current_characteristics()
            }
            
            current_phase = self.current_cycle.get_current_phase()
            if current_phase:
                status['current_phase'] = {
                    'type': current_phase.phase_type.value,
                    'progress': current_phase.current_progress,
                    'energy': current_phase.get_current_energy(),
                    'stability': current_phase.get_current_stability()
                }
        
        # Add Antikythera status
        if self.antikythera_orchestrator:
            antikythera_status = self.antikythera_orchestrator.get_current_status()
            status['antikythera'] = {
                'is_running': antikythera_status['is_running'],
                'simulation_time': antikythera_status['simulation_time'],
                'cycles': antikythera_status['cycles']
            }
        
        # Add evolutionary status
        if self.evolutionary_orchestrator:
            evo_status = self.evolutionary_orchestrator.get_current_status()
            status['evolution'] = {
                'is_running': evo_status['is_running'],
                'populations': evo_status['populations'],
                'best_fitness': evo_status.get('best_fitness', 0.0)
            }
        
        # Add civilizational engine status
        status['civilizational_engine'] = self.civilizational_engine.get_status()
        
        return status
    
    def _count_completed_cycles(self) -> int:
        """Count completed civilizational cycles."""
        if not self.antikythera_orchestrator:
            return 0
        
        completed = 0
        for cycle in self.antikythera_orchestrator.meta_cycle.cycles:
            if cycle.completed:
                completed += 1
        
        return completed
    
    def _get_evolutionary_results(self) -> Dict[str, Any]:
        """Get results from evolutionary computation."""
        if not self.evolutionary_orchestrator:
            return {}
        
        status = self.evolutionary_orchestrator.get_current_status()
        
        return {
            'populations': status.get('populations', 0),
            'best_fitness': status.get('best_fitness', 0.0),
            'total_generations': sum(
                engine.generation for engine in self.evolutionary_orchestrator.engines
                if hasattr(engine, 'generation')
            )
        }
    
    def register_transformation_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for transformation events.
        
        Args:
            callback: Function to call on transformation events
        """
        self.transformation_callbacks.append(callback)
    
    def register_cycle_completion_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for cycle completion.
        
        Args:
            callback: Function to call when cycles complete
        """
        self.cycle_completion_callbacks.append(callback)
    
    def save_transformation_state(self, filepath: str) -> None:
        """Save complete transformation state to file.
        
        Args:
            filepath: Path to save state file
        """
        state = {
            'config': self.config.__dict__,
            'transformation_status': self.get_transformation_status(),
            'transformation_history': self.transformation_history,
            'civilizational_state': self.civilizational_engine.get_state()
        }
        
        # Add component states
        if self.antikythera_orchestrator:
            state['antikythera_state'] = self.antikythera_orchestrator.get_current_status()
        
        if self.evolutionary_orchestrator:
            state['evolutionary_state'] = self.evolutionary_orchestrator.get_current_status()
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Transformation state saved to {filepath}")
    
    def predict_future_transformation(self, years_ahead: int = 100) -> Dict[str, Any]:
        """Predict future transformation trajectory.
        
        Args:
            years_ahead: Years to predict ahead
            
        Returns:
            Prediction results
        """
        if self.antikythera_orchestrator:
            return self.antikythera_orchestrator.predict_future_trajectory(years_ahead)
        
        return {'error': 'Antikythera orchestrator not available for prediction'}


class CycleOfPhoenix:
    """
    Main interface class for the Cycle of Phoenix framework.
    
    This provides a simplified interface to the complete Phoenix transformation
    system, making it easy to use for various applications.
    """
    
    def __init__(self, atomspace: AtomSpace, 
                 transformation_mode: TransformationMode = TransformationMode.CONSCIOUSNESS_EVOLUTION):
        """Initialize the Cycle of Phoenix framework.
        
        Args:
            atomspace: The AtomSpace for cognitive operations
            transformation_mode: Mode of transformation to use
        """
        self.atomspace = atomspace
        self.transformation_mode = transformation_mode
        
        # Create configuration
        config = PhoenixConfiguration(
            transformation_mode=transformation_mode,
            time_acceleration=1000.0,  # Fast simulation
            population_size=50,        # Moderate population
            max_generations=200        # Reasonable evolution
        )
        
        # Create orchestrator
        self.orchestrator = PhoenixOrchestrator(atomspace, config)
        
        # Setup default monitoring
        self._setup_default_monitoring()
    
    def _setup_default_monitoring(self) -> None:
        """Setup default monitoring and logging."""
        def log_transformation_events(event_info: Dict[str, Any]) -> None:
            """Log transformation events."""
            phase = event_info.get('phase', 'unknown')
            cycle_id = event_info.get('cycle_id', 0)
            logger.info(f"Phoenix transformation: {phase} in cycle {cycle_id}")
        
        def log_cycle_completion(completion_info: Dict[str, Any]) -> None:
            """Log cycle completion."""
            cycles_completed = completion_info.get('cycles_completed', 0)
            runtime = completion_info.get('total_runtime', 0)
            logger.info(f"Phoenix cycles completed: {cycles_completed} in {runtime:.2f} seconds")
        
        self.orchestrator.register_transformation_callback(log_transformation_events)
        self.orchestrator.register_cycle_completion_callback(log_cycle_completion)
    
    def begin_cycle_of_transformation(self) -> None:
        """Begin the Cycle of Phoenix transformation."""
        self.orchestrator.begin_cycle_of_transformation()
    
    def get_transformation_status(self) -> Dict[str, Any]:
        """Get current transformation status."""
        return self.orchestrator.get_transformation_status()
    
    def pause_transformation(self) -> None:
        """Pause the transformation."""
        self.orchestrator.pause_transformation()
    
    def resume_transformation(self) -> None:
        """Resume the transformation."""
        self.orchestrator.resume_transformation()
    
    def complete_transformation(self) -> Dict[str, Any]:
        """Complete the transformation and get results."""
        return self.orchestrator.complete_transformation()
    
    def predict_future(self, years_ahead: int = 100) -> Dict[str, Any]:
        """Predict future transformation trajectory."""
        return self.orchestrator.predict_future_transformation(years_ahead)
    
    def save_state(self, filepath: str) -> None:
        """Save transformation state to file."""
        self.orchestrator.save_transformation_state(filepath)