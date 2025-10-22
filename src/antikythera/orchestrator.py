"""
Antikythera Orchestration Framework

This module implements the main orchestrator for the Antikythera framework,
coordinating civilizational cycles with cognitive evolution using the MOSES engine.
The orchestrator acts as the central coordination mechanism, like the gears of 
the ancient Antikythera mechanism.
"""

import time
import threading
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from .cycles import CivilizationalCycle, MetaCycle, PhaseType
from .temporal import TemporalCoordinator, TimeScale
from .integration import CognitiveIntegration, EvolutionaryIntegration
from ..atomspace import AtomSpace, Node, Link, TruthValue, AttentionValue
from ..evolution import EvolutionaryOrchestrator, MosesEngine, Individual

# Configure logging
logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Modes of orchestration operation."""
    HISTORICAL_SIMULATION = "historical_simulation"
    PREDICTIVE_MODELING = "predictive_modeling"
    REAL_TIME_ADAPTATION = "real_time_adaptation"
    COGNITIVE_EVOLUTION = "cognitive_evolution"


@dataclass
class OrchestrationConfig:
    """Configuration for the Antikythera orchestrator."""
    mode: OrchestrationMode = OrchestrationMode.COGNITIVE_EVOLUTION
    time_acceleration: float = 1.0  # Time acceleration factor
    enable_evolution: bool = True
    enable_cognitive_integration: bool = True
    save_state_interval: int = 100  # Save state every N updates
    max_cycles: int = 10
    cycle_interaction_threshold: float = 0.1
    
    # Evolution integration parameters
    evolution_sync_interval: int = 10
    fitness_adaptation_rate: float = 0.1
    
    # Cognitive integration parameters
    atomspace_sync_interval: int = 5
    attention_decay_rate: float = 0.95


class AntikytheraOrchestrator:
    """
    Main orchestrator for the Antikythera framework.
    
    This class coordinates civilizational cycles, cognitive evolution, and temporal
    patterns to create a comprehensive framework for understanding and modeling
    complex transformational dynamics.
    """
    
    def __init__(self, atomspace: AtomSpace, config: OrchestrationConfig = None):
        """Initialize the Antikythera orchestrator.
        
        Args:
            atomspace: The AtomSpace for cognitive operations
            config: Configuration for orchestration
        """
        self.atomspace = atomspace
        self.config = config or OrchestrationConfig()
        
        # Core components
        self.meta_cycle = MetaCycle("antikythera_meta_cycle")
        self.temporal_coordinator = TemporalCoordinator()
        
        # Integration modules
        self.cognitive_integration = CognitiveIntegration(atomspace)
        self.evolutionary_integration: Optional[EvolutionaryIntegration] = None
        
        # Orchestration state
        self.is_running = False
        self.start_time: Optional[float] = None
        self.simulation_time = 0.0  # Simulated time in years
        self.update_count = 0
        
        # Threading for async operation
        self.orchestration_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        
        # Event callbacks
        self.cycle_callbacks: List[Callable] = []
        self.phase_callbacks: List[Callable] = []
        self.evolution_callbacks: List[Callable] = []
        
        # State persistence
        self.state_history: List[Dict[str, Any]] = []
        self.last_save_time = 0.0
        
        # Initialize with default cycles if in historical mode
        if self.config.mode == OrchestrationMode.HISTORICAL_SIMULATION:
            self._initialize_historical_cycles()
    
    def _initialize_historical_cycles(self) -> None:
        """Initialize with the six historical cycles from the Cycle of the Phoenix."""
        historical_cycles = [
            ("Bronze Age to Axial Age", -1200),
            ("Classical Period", -600), 
            ("Roman to Medieval", 0),
            ("Medieval Period", 600),
            ("Late Medieval to Modern", 1200),
            ("Modern to Future", 1800)
        ]
        
        for i, (name, start_year) in enumerate(historical_cycles):
            cycle = CivilizationalCycle(name, cycle_id=i)
            
            # Bind to cognitive system
            cycle.bind_to_cognitive_system(self.atomspace)
            
            self.meta_cycle.add_cycle(cycle)
            
            logger.info(f"Added historical cycle: {name} (starting {start_year})")
        
        # Set up cycle interactions (simplified model)
        for i in range(len(historical_cycles) - 1):
            # Each cycle influences the next one
            self.meta_cycle.set_cycle_interaction(i, i + 1, 0.3)
    
    def add_cycle(self, cycle: CivilizationalCycle) -> None:
        """Add a civilizational cycle to the orchestration.
        
        Args:
            cycle: The cycle to add
        """
        cycle.bind_to_cognitive_system(self.atomspace)
        self.meta_cycle.add_cycle(cycle)
        
        # Bind to evolution if available
        if self.evolutionary_integration:
            engine = self.evolutionary_integration.get_primary_engine()
            if engine:
                cycle.bind_to_evolution(engine)
        
        logger.info(f"Added cycle: {cycle.name}")
    
    def integrate_evolution(self, evolutionary_orchestrator: EvolutionaryOrchestrator) -> None:
        """Integrate with evolutionary orchestration.
        
        Args:
            evolutionary_orchestrator: The evolutionary orchestrator to integrate
        """
        self.evolutionary_integration = EvolutionaryIntegration(
            evolutionary_orchestrator, 
            self.atomspace
        )
        
        # Bind existing cycles to evolution
        for cycle in self.meta_cycle.cycles:
            engine = self.evolutionary_integration.get_primary_engine()
            if engine:
                cycle.bind_to_evolution(engine)
        
        logger.info("Integrated evolutionary orchestration")
    
    def start_orchestration(self, async_mode: bool = True) -> None:
        """Start the orchestration process.
        
        Args:
            async_mode: Whether to run in asynchronous mode
        """
        if self.is_running:
            logger.warning("Orchestration is already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.simulation_time = 0.0
        self.update_count = 0
        self.should_stop.clear()
        
        # Start meta-cycle
        self.meta_cycle.start_meta_cycle()
        
        # Start temporal coordination
        self.temporal_coordinator.start()
        
        # Start evolutionary integration if available
        if (self.evolutionary_integration and 
            self.config.enable_evolution):
            self.evolutionary_integration.start_integration()
        
        if async_mode:
            self.orchestration_thread = threading.Thread(
                target=self._orchestration_loop,
                name="AntikytheraOrchestrator"
            )
            self.orchestration_thread.start()
            logger.info("Started Antikythera orchestration (async mode)")
        else:
            self._orchestration_loop()
    
    def stop_orchestration(self) -> None:
        """Stop the orchestration process."""
        if not self.is_running:
            return
        
        logger.info("Stopping Antikythera orchestration")
        
        self.should_stop.set()
        self.is_running = False
        
        # Stop evolutionary integration
        if self.evolutionary_integration:
            self.evolutionary_integration.stop_integration()
        
        # Wait for orchestration thread to finish
        if self.orchestration_thread and self.orchestration_thread.is_alive():
            self.orchestration_thread.join(timeout=5.0)
        
        # Save final state
        self._save_current_state()
        
        logger.info("Antikythera orchestration stopped")
    
    def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        logger.info("Starting orchestration loop")
        
        try:
            while self.is_running and not self.should_stop.is_set():
                # Calculate time step
                current_time = time.time()
                if self.start_time:
                    real_elapsed = current_time - self.start_time
                    # Convert real time to simulation time with acceleration
                    time_step = real_elapsed * self.config.time_acceleration
                    self.simulation_time = time_step
                
                # Update all cycles
                self._update_cycles()
                
                # Update temporal coordination
                self._update_temporal_coordination()
                
                # Update cognitive integration
                if self.config.enable_cognitive_integration:
                    self._update_cognitive_integration()
                
                # Update evolutionary integration
                if (self.evolutionary_integration and 
                    self.config.enable_evolution):
                    self._update_evolutionary_integration()
                
                # Trigger callbacks
                self._trigger_update_callbacks()
                
                # Save state periodically
                if (self.update_count % self.config.save_state_interval == 0):
                    self._save_current_state()
                
                self.update_count += 1
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in orchestration loop: {e}")
            raise
        finally:
            logger.info("Orchestration loop ended")
    
    def _update_cycles(self) -> None:
        """Update all civilizational cycles."""
        self.meta_cycle.update_all_cycles(self.simulation_time)
        
        # Check for phase transitions and trigger callbacks
        for cycle in self.meta_cycle.cycles:
            if cycle.is_active:
                current_phase = cycle.get_current_phase()
                if current_phase:
                    # Apply evolutionary pressure based on cycle phase
                    cycle.apply_evolutionary_pressure()
    
    def _update_temporal_coordination(self) -> None:
        """Update temporal coordination."""
        # Get current temporal patterns from all active cycles
        temporal_patterns = []
        
        for cycle in self.meta_cycle.cycles:
            if cycle.is_active:
                current_phase = cycle.get_current_phase()
                if current_phase:
                    # Create temporal pattern from phase characteristics
                    pattern_data = {
                        'cycle_id': cycle.cycle_id,
                        'phase_type': current_phase.phase_type.value,
                        'energy': current_phase.get_current_energy(),
                        'stability': current_phase.get_current_stability(),
                        'progress': current_phase.current_progress
                    }
                    temporal_patterns.append(pattern_data)
        
        # Update temporal coordinator with patterns
        self.temporal_coordinator.update_patterns(temporal_patterns)
    
    def _update_cognitive_integration(self) -> None:
        """Update cognitive integration with AtomSpace."""
        if self.update_count % self.config.atomspace_sync_interval == 0:
            # Get aggregate characteristics from all cycles
            meta_chars = self.meta_cycle.get_meta_characteristics()
            
            # Update cognitive integration
            self.cognitive_integration.update_from_cycles(meta_chars)
            
            # Update attention values based on cycle dynamics
            self._update_attention_dynamics()
    
    def _update_evolutionary_integration(self) -> None:
        """Update evolutionary integration."""
        if (self.evolutionary_integration and 
            self.update_count % self.config.evolution_sync_interval == 0):
            
            # Get current civilizational characteristics
            meta_chars = self.meta_cycle.get_meta_characteristics()
            
            # Update evolutionary parameters based on civilizational state
            self.evolutionary_integration.adapt_evolution_parameters(meta_chars)
            
            # Synchronize evolutionary progress with cycle dynamics
            self.evolutionary_integration.sync_with_cycles(self.meta_cycle)
    
    def _update_attention_dynamics(self) -> None:
        """Update attention dynamics in AtomSpace based on cycle state."""
        # This implements the attention economy influenced by civilizational cycles
        
        for cycle in self.meta_cycle.cycles:
            if not cycle.is_active:
                continue
            
            current_phase = cycle.get_current_phase()
            if not current_phase:
                continue
            
            # Get cycle node from AtomSpace
            cycle_node_name = f"civilizational_cycle_{cycle.name}"
            cycle_atoms = self.atomspace.get_atoms_by_type("ConceptNode")
            
            for atom in cycle_atoms:
                if hasattr(atom, 'name') and atom.name == cycle_node_name:
                    # Update attention based on phase characteristics
                    energy = current_phase.get_current_energy()
                    stability = current_phase.get_current_stability()
                    
                    # Short-term importance based on energy
                    sti = energy * 100.0 - 50.0  # Range: -50 to +50
                    
                    # Long-term importance based on stability
                    lti = stability * 50.0  # Range: 0 to 50
                    
                    # Update attention value
                    av = AttentionValue(sti=sti, lti=lti)
                    atom.set_attention_value(av)
    
    def _trigger_update_callbacks(self) -> None:
        """Trigger registered callbacks for updates."""
        update_info = {
            'simulation_time': self.simulation_time,
            'update_count': self.update_count,
            'active_cycles': len([c for c in self.meta_cycle.cycles if c.is_active]),
            'meta_characteristics': self.meta_cycle.get_meta_characteristics()
        }
        
        for callback in self.cycle_callbacks:
            try:
                callback(update_info)
            except Exception as e:
                logger.error(f"Error in cycle callback: {e}")
    
    def _save_current_state(self) -> None:
        """Save current orchestration state."""
        state = {
            'timestamp': time.time(),
            'simulation_time': self.simulation_time,
            'update_count': self.update_count,
            'meta_characteristics': self.meta_cycle.get_meta_characteristics(),
            'active_cycles': []
        }
        
        # Save state of active cycles
        for cycle in self.meta_cycle.cycles:
            if cycle.is_active:
                cycle_state = {
                    'cycle_id': cycle.cycle_id,
                    'name': cycle.name,
                    'progress': cycle.get_cycle_progress(),
                    'current_phase': cycle.current_phase_index,
                    'characteristics': cycle.get_current_characteristics()
                }
                
                current_phase = cycle.get_current_phase()
                if current_phase:
                    cycle_state['phase_info'] = {
                        'type': current_phase.phase_type.value,
                        'progress': current_phase.current_progress,
                        'energy': current_phase.get_current_energy(),
                        'stability': current_phase.get_current_stability()
                    }
                
                state['active_cycles'].append(cycle_state)
        
        self.state_history.append(state)
        self.last_save_time = time.time()
    
    def save_full_state(self, filepath: str) -> None:
        """Save complete orchestrator state to file.
        
        Args:
            filepath: Path to save state file
        """
        full_state = {
            'config': self.config.__dict__,
            'simulation_time': self.simulation_time,
            'update_count': self.update_count,
            'state_history': self.state_history,
            'meta_cycle_state': {
                'name': self.meta_cycle.name,
                'cycle_count': len(self.meta_cycle.cycles),
                'interactions': list(self.meta_cycle.cycle_interactions.items())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_state, f, indent=2, default=str)
        
        logger.info(f"Full orchestrator state saved to {filepath}")
    
    def load_full_state(self, filepath: str) -> None:
        """Load complete orchestrator state from file.
        
        Args:
            filepath: Path to state file
        """
        with open(filepath, 'r') as f:
            full_state = json.load(f)
        
        # Restore configuration
        config_dict = full_state.get('config', {})
        self.config = OrchestrationConfig(**config_dict)
        
        # Restore state
        self.simulation_time = full_state.get('simulation_time', 0.0)
        self.update_count = full_state.get('update_count', 0)
        self.state_history = full_state.get('state_history', [])
        
        logger.info(f"Full orchestrator state loaded from {filepath}")
    
    def register_cycle_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for cycle updates.
        
        Args:
            callback: Function to call on cycle updates
        """
        self.cycle_callbacks.append(callback)
    
    def register_phase_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """Register callback for phase transitions.
        
        Args:
            callback: Function to call on phase transitions
        """
        self.phase_callbacks.append(callback)
    
    def register_evolution_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for evolution events.
        
        Args:
            callback: Function to call on evolution events
        """
        self.evolution_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current orchestration status.
        
        Returns:
            Dictionary containing current status information
        """
        status = {
            'is_running': self.is_running,
            'simulation_time': self.simulation_time,
            'update_count': self.update_count,
            'mode': self.config.mode.value,
            'cycles': {
                'total': len(self.meta_cycle.cycles),
                'active': len([c for c in self.meta_cycle.cycles if c.is_active]),
                'completed': len([c for c in self.meta_cycle.cycles if c.completed])
            },
            'meta_characteristics': self.meta_cycle.get_meta_characteristics(),
            'integrations': {
                'cognitive': self.config.enable_cognitive_integration,
                'evolutionary': self.evolutionary_integration is not None and self.config.enable_evolution
            }
        }
        
        # Add detailed cycle information
        status['cycle_details'] = []
        for cycle in self.meta_cycle.cycles:
            cycle_info = {
                'id': cycle.cycle_id,
                'name': cycle.name,
                'is_active': cycle.is_active,
                'completed': cycle.completed,
                'progress': cycle.get_cycle_progress()
            }
            
            if cycle.is_active:
                current_phase = cycle.get_current_phase()
                if current_phase:
                    cycle_info['current_phase'] = {
                        'type': current_phase.phase_type.value,
                        'progress': current_phase.current_progress,
                        'characteristics': cycle.get_current_characteristics()
                    }
            
            status['cycle_details'].append(cycle_info)
        
        return status
    
    def get_cycle_by_name(self, name: str) -> Optional[CivilizationalCycle]:
        """Get a cycle by name.
        
        Args:
            name: Name of the cycle to find
            
        Returns:
            The cycle if found, None otherwise
        """
        for cycle in self.meta_cycle.cycles:
            if cycle.name == name:
                return cycle
        return None
    
    def get_historical_context(self, year: int) -> Optional[Dict[str, Any]]:
        """Get historical context for a specific year.
        
        Args:
            year: Year to get context for
            
        Returns:
            Historical context information
        """
        # Try to get context from any cycle
        for cycle in self.meta_cycle.cycles:
            context = cycle.get_historical_context(year)
            if context:
                return context
        
        return None
    
    def simulate_historical_period(self, start_year: int, end_year: int, 
                                 time_step: float = 1.0) -> List[Dict[str, Any]]:
        """Simulate a historical period and return results.
        
        Args:
            start_year: Starting year for simulation
            end_year: Ending year for simulation  
            time_step: Time step for simulation (in years)
            
        Returns:
            List of simulation states
        """
        if self.is_running:
            raise RuntimeError("Cannot run historical simulation while orchestration is active")
        
        simulation_results = []
        current_year = start_year
        
        # Temporarily set historical simulation mode
        original_mode = self.config.mode
        self.config.mode = OrchestrationMode.HISTORICAL_SIMULATION
        
        try:
            # Initialize for historical simulation
            if not self.meta_cycle.cycles:
                self._initialize_historical_cycles()
            
            # Set up initial state
            self.simulation_time = 0.0
            self.meta_cycle.start_meta_cycle()
            
            while current_year <= end_year:
                # Update cycles
                elapsed_time = current_year - start_year
                self.meta_cycle.update_all_cycles(elapsed_time)
                
                # Record state
                state = {
                    'year': current_year,
                    'simulation_time': elapsed_time,
                    'meta_characteristics': self.meta_cycle.get_meta_characteristics(),
                    'active_cycles': []
                }
                
                for cycle in self.meta_cycle.cycles:
                    if cycle.is_active:
                        cycle_state = {
                            'name': cycle.name,
                            'progress': cycle.get_cycle_progress(),
                            'characteristics': cycle.get_current_characteristics()
                        }
                        
                        current_phase = cycle.get_current_phase()
                        if current_phase:
                            cycle_state['phase'] = {
                                'type': current_phase.phase_type.value,
                                'progress': current_phase.current_progress
                            }
                        
                        state['active_cycles'].append(cycle_state)
                
                # Get historical context
                historical_context = self.get_historical_context(int(current_year))
                if historical_context:
                    state['historical_context'] = historical_context
                
                simulation_results.append(state)
                current_year += time_step
            
        finally:
            # Restore original mode
            self.config.mode = original_mode
        
        return simulation_results
    
    def predict_future_trajectory(self, years_ahead: int = 200) -> Dict[str, Any]:
        """Predict future civilizational trajectory.
        
        Args:
            years_ahead: Number of years to predict ahead
            
        Returns:
            Prediction results
        """
        current_characteristics = self.meta_cycle.get_meta_characteristics()
        
        # Simple predictive model based on current trends
        # In a full implementation, this would use more sophisticated modeling
        
        prediction = {
            'prediction_horizon': years_ahead,
            'current_state': current_characteristics,
            'predicted_trajectory': [],
            'confidence_intervals': {},
            'key_transitions': []
        }
        
        # Generate trajectory points
        for year_offset in range(0, years_ahead, 10):
            # Simple linear extrapolation with some cyclical components
            progress_factor = year_offset / years_ahead
            
            predicted_chars = {}
            for key, current_value in current_characteristics.items():
                # Add some cyclical variation
                cycle_component = 0.1 * math.sin(2 * math.pi * progress_factor)
                trend_component = -0.1 * progress_factor  # Slight decline trend
                
                predicted_value = current_value + trend_component + cycle_component
                predicted_chars[key] = max(0.0, min(1.0, predicted_value))
            
            prediction['predicted_trajectory'].append({
                'year_offset': year_offset,
                'characteristics': predicted_chars
            })
        
        return prediction


# Import needed for the prediction method
import math