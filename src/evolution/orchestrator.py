"""
Evolutionary Orchestrator for MOSES Engine

This module provides high-level orchestration of evolutionary processes,
managing multiple evolutionary runs, adaptive parameter adjustment, 
and integration with the CogPrime cognitive architecture.
"""

import time
import threading
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, Future

from .moses_engine import MosesEngine, Individual, EvolutionHistory
from .population import Population, PopulationInitializer, RandomInitializer
from .selection import SelectionStrategy, TournamentSelection, AdaptiveSelection
from src.atomspace import AtomSpace, Atom, Node, Link

# Configure logging
logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Different modes of evolutionary orchestration."""
    SINGLE_POPULATION = "single_population"
    MULTI_POPULATION = "multi_population" 
    ISLAND_MODEL = "island_model"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary runs."""
    population_size: int = 50
    max_generations: int = 100
    target_fitness: float = 0.95
    stagnation_limit: int = 20
    migration_rate: float = 0.1
    migration_interval: int = 10
    elite_preservation: float = 0.1
    diversity_threshold: float = 0.2
    adaptive_parameters: bool = True
    parallel_evaluation: bool = True
    save_history: bool = True


@dataclass
class EvolutionResult:
    """Result from an evolutionary run."""
    best_individual: Individual
    final_fitness: float
    generations: int
    convergence_time: float
    population_stats: Dict[str, Any]
    history: Optional[EvolutionHistory] = None
    success: bool = False
    termination_reason: str = ""


class EvolutionaryOrchestrator:
    """Orchestrates complex evolutionary processes for cognitive architecture."""
    
    def __init__(self, atomspace: AtomSpace, 
                 config: EvolutionConfig = None,
                 mode: OrchestrationMode = OrchestrationMode.ADAPTIVE):
        """Initialize evolutionary orchestrator.
        
        Args:
            atomspace: The AtomSpace for cognitive structure management
            config: Configuration for evolution
            mode: Orchestration mode
        """
        self.atomspace = atomspace
        self.config = config or EvolutionConfig()
        self.mode = mode
        
        # Evolution engines and populations
        self.engines: List[MosesEngine] = []
        self.populations: List[Population] = []
        self.histories: List[EvolutionHistory] = []
        
        # Orchestration state
        self.is_running = False
        self.current_generation = 0
        self.start_time: Optional[float] = None
        self.best_overall: Optional[Individual] = None
        
        # Threading for parallel execution
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.evolution_thread: Optional[threading.Thread] = None
        
        # Callbacks and monitoring
        self.generation_callbacks: List[Callable] = []
        self.convergence_callbacks: List[Callable] = []
        self.migration_callbacks: List[Callable] = []
        
        # Adaptive parameter management
        self.parameter_history: List[Dict[str, Any]] = []
        self.adaptation_interval = 10
        
    def add_population(self, size: int = None, 
                      initializer: PopulationInitializer = None,
                      selection_strategy: SelectionStrategy = None) -> int:
        """Add a population to the orchestrator.
        
        Args:
            size: Population size (uses config default if None)
            initializer: Population initializer
            selection_strategy: Selection strategy for this population
            
        Returns:
            Index of the added population
        """
        pop_size = size or self.config.population_size
        
        # Create evolution engine
        engine = MosesEngine(self.atomspace, pop_size)
        
        # Set selection strategy if provided
        if selection_strategy is not None:
            # Note: This would require extending MosesEngine to support custom selection
            pass
        
        # Create population with initializer
        if initializer is None:
            # Create default initializer with existing atoms
            all_atoms = self.atomspace.get_all_atoms()
            if not all_atoms:
                # Create some basic seed atoms
                all_atoms = [
                    Node("ConceptNode", "knowledge"),
                    Node("ConceptNode", "learning"), 
                    Node("ConceptNode", "adaptation"),
                    Node("PredicateNode", "relevance"),
                    Link("InheritanceLink", [
                        Node("ConceptNode", "specific"),
                        Node("ConceptNode", "general")
                    ])
                ]
                for atom in all_atoms:
                    self.atomspace.add(atom)
            
            initializer = RandomInitializer(all_atoms, min_genes=1, max_genes=5)
        
        population = Population(initializer=initializer)
        
        # Initialize population through engine
        engine.initialize_population(all_atoms)
        population.individuals = engine.population
        
        # Add to orchestrator
        self.engines.append(engine)
        self.populations.append(population)
        self.histories.append(EvolutionHistory())
        
        return len(self.populations) - 1
    
    def remove_population(self, index: int) -> bool:
        """Remove a population from the orchestrator.
        
        Args:
            index: Index of population to remove
            
        Returns:
            True if population was removed successfully
        """
        if 0 <= index < len(self.populations):
            self.engines.pop(index)
            self.populations.pop(index) 
            self.histories.pop(index)
            return True
        return False
    
    def start_evolution(self, async_mode: bool = False) -> Optional[Future]:
        """Start the evolutionary process.
        
        Args:
            async_mode: If True, run evolution in background thread
            
        Returns:
            Future object if async_mode is True, None otherwise
        """
        if self.is_running:
            logger.warning("Evolution is already running")
            return None
        
        if not self.populations:
            logger.error("No populations added for evolution")
            return None
        
        self.is_running = True
        self.start_time = time.time()
        self.current_generation = 0
        
        if async_mode:
            if self.thread_pool is None:
                self.thread_pool = ThreadPoolExecutor(max_workers=len(self.populations))
            
            future = self.thread_pool.submit(self._run_evolution)
            return future
        else:
            return self._run_evolution()
    
    def stop_evolution(self) -> None:
        """Stop the evolutionary process."""
        self.is_running = False
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
    
    def _run_evolution(self) -> EvolutionResult:
        """Main evolution loop."""
        logger.info(f"Starting evolution with {len(self.populations)} populations")
        
        try:
            while (self.is_running and 
                   self.current_generation < self.config.max_generations):
                
                # Evolve all populations
                self._evolve_generation()
                
                # Check convergence criteria
                if self._check_convergence():
                    logger.info(f"Convergence achieved at generation {self.current_generation}")
                    break
                
                # Perform migration if needed
                if (self.current_generation % self.config.migration_interval == 0 and
                    len(self.populations) > 1):
                    self._perform_migration()
                
                # Adaptive parameter adjustment
                if (self.config.adaptive_parameters and 
                    self.current_generation % self.adaptation_interval == 0):
                    self._adapt_parameters()
                
                # Trigger generation callbacks
                self._trigger_generation_callbacks()
                
                self.current_generation += 1
                time.sleep(0.01)  # Small delay to prevent CPU overload
            
            # Create final result
            result = self._create_final_result()
            
            # Trigger convergence callbacks
            for callback in self.convergence_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in convergence callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during evolution: {e}")
            self.is_running = False
            raise
        finally:
            self.is_running = False
    
    def _evolve_generation(self) -> None:
        """Evolve one generation across all populations."""
        if self.config.parallel_evaluation and len(self.populations) > 1:
            # Parallel evolution
            if self.thread_pool is None:
                self.thread_pool = ThreadPoolExecutor(max_workers=len(self.populations))
            
            # Submit evolution tasks
            futures = []
            for i, engine in enumerate(self.engines):
                future = self.thread_pool.submit(engine.evolve_generation)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
        else:
            # Sequential evolution
            for engine in self.engines:
                engine.evolve_generation()
        
        # Update population records and histories
        for i, (engine, population, history) in enumerate(
            zip(self.engines, self.populations, self.histories)):
            
            # Sync engine population with population object
            population.individuals = engine.population
            population.generation = engine.generation
            
            # Record history
            history.record_generation(engine)
            
            # Update best overall individual
            best_in_pop = engine.get_best_individual()
            if (best_in_pop and 
                (self.best_overall is None or 
                 (best_in_pop.fitness or 0.0) > (self.best_overall.fitness or 0.0))):
                self.best_overall = best_in_pop.copy()
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if self.best_overall is None:
            return False
        
        # Check target fitness
        if (self.best_overall.fitness or 0.0) >= self.config.target_fitness:
            return True
        
        # Check stagnation across all populations
        all_stagnant = True
        for history in self.histories:
            convergence_analysis = history.analyze_convergence()
            if not convergence_analysis.get('converged', False):
                all_stagnant = False
                break
        
        return all_stagnant
    
    def _perform_migration(self) -> None:
        """Perform migration between populations."""
        if len(self.populations) < 2:
            return
        
        logger.debug(f"Performing migration at generation {self.current_generation}")
        
        migrants_per_pop = max(1, int(self.config.population_size * self.config.migration_rate))
        
        # Collect migrants from each population (best individuals)
        migrants = []
        for population in self.populations:
            best_individuals = population.get_best(migrants_per_pop)
            migrants.append(best_individuals)
        
        # Redistribute migrants (circular migration)
        for i, population in enumerate(self.populations):
            # Get migrants from next population (circular)
            source_pop_idx = (i + 1) % len(self.populations)
            incoming_migrants = migrants[source_pop_idx]
            
            # Replace worst individuals with migrants
            worst_individuals = population.get_worst(len(incoming_migrants))
            
            # Replace in population
            for j, migrant in enumerate(incoming_migrants):
                if j < len(worst_individuals):
                    # Find and replace worst individual
                    for k, individual in enumerate(population.individuals):
                        if individual == worst_individuals[j]:
                            population.individuals[k] = migrant.copy()
                            population.individuals[k].generation = self.current_generation
                            break
        
        # Trigger migration callbacks
        for callback in self.migration_callbacks:
            try:
                callback(self.current_generation, migrants)
            except Exception as e:
                logger.error(f"Error in migration callback: {e}")
    
    def _adapt_parameters(self) -> None:
        """Adaptively adjust evolution parameters."""
        # Collect current statistics
        current_stats = {
            'generation': self.current_generation,
            'populations': []
        }
        
        for i, (engine, history) in enumerate(zip(self.engines, self.histories)):
            pop_stats = engine.get_population_stats()
            convergence = history.analyze_convergence()
            
            current_stats['populations'].append({
                'index': i,
                'stats': pop_stats,
                'convergence': convergence
            })
        
        self.parameter_history.append(current_stats)
        
        # Adapt parameters based on trends
        if len(self.parameter_history) >= 3:
            self._adjust_mutation_rates()
            self._adjust_selection_pressure()
            self._adjust_population_sizes()
    
    def _adjust_mutation_rates(self) -> None:
        """Adjust mutation rates based on diversity trends."""
        for i, engine in enumerate(self.engines):
            # Get diversity trend
            recent_stats = [stats['populations'][i]['stats'] 
                          for stats in self.parameter_history[-3:]]
            
            diversity_trend = (recent_stats[-1]['diversity'] - 
                             recent_stats[0]['diversity']) / 3
            
            # Adjust mutation rate based on diversity trend
            for operator in engine.operators:
                if hasattr(operator, 'mutation_rate'):
                    current_rate = operator.mutation_rate
                    
                    if diversity_trend < -0.01:  # Diversity decreasing
                        operator.mutation_rate = min(0.5, current_rate * 1.1)
                    elif diversity_trend > 0.01:  # Diversity increasing
                        operator.mutation_rate = max(0.01, current_rate * 0.9)
    
    def _adjust_selection_pressure(self) -> None:
        """Adjust selection pressure based on convergence."""
        for i, engine in enumerate(self.engines):
            convergence_info = self.parameter_history[-1]['populations'][i]['convergence']
            
            if convergence_info.get('converged', False):
                # Increase selection pressure if converged
                engine.selection_pressure = min(3.0, engine.selection_pressure * 1.1)
            else:
                # Decrease selection pressure if not converged
                engine.selection_pressure = max(1.0, engine.selection_pressure * 0.95)
    
    def _adjust_population_sizes(self) -> None:
        """Adjust population sizes based on performance."""
        # This is a simplified version - in practice, this would be more sophisticated
        total_performance = 0.0
        performances = []
        
        for i, stats in enumerate(self.parameter_history[-1]['populations']):
            performance = stats['stats'].get('average_fitness', 0.0)
            performances.append(performance)
            total_performance += performance
        
        if total_performance > 0:
            # Redistribute population sizes based on relative performance
            base_size = self.config.population_size
            
            for i, performance in enumerate(performances):
                relative_performance = performance / total_performance * len(performances)
                new_size = int(base_size * (0.5 + 0.5 * relative_performance))
                new_size = max(10, min(100, new_size))  # Bounds
                
                # Note: Actual population resizing would require more implementation
                logger.debug(f"Population {i} performance ratio: {relative_performance:.3f}")
    
    def _trigger_generation_callbacks(self) -> None:
        """Trigger callbacks for generation completion."""
        generation_info = {
            'generation': self.current_generation,
            'best_fitness': self.best_overall.fitness if self.best_overall else 0.0,
            'population_stats': [engine.get_population_stats() for engine in self.engines],
            'runtime': time.time() - (self.start_time or 0)
        }
        
        for callback in self.generation_callbacks:
            try:
                callback(generation_info)
            except Exception as e:
                logger.error(f"Error in generation callback: {e}")
    
    def _create_final_result(self) -> EvolutionResult:
        """Create the final evolution result."""
        runtime = time.time() - (self.start_time or 0)
        
        # Determine success and termination reason
        success = False
        termination_reason = "max_generations_reached"
        
        if self.best_overall and (self.best_overall.fitness or 0.0) >= self.config.target_fitness:
            success = True
            termination_reason = "target_fitness_achieved"
        elif self._check_convergence():
            success = True
            termination_reason = "convergence_detected"
        elif not self.is_running:
            termination_reason = "manually_stopped"
        
        # Aggregate population stats
        all_pop_stats = {
            'total_populations': len(self.populations),
            'total_individuals': sum(len(pop) for pop in self.populations),
            'population_details': [engine.get_population_stats() for engine in self.engines]
        }
        
        return EvolutionResult(
            best_individual=self.best_overall or Individual(),
            final_fitness=self.best_overall.fitness if self.best_overall else 0.0,
            generations=self.current_generation,
            convergence_time=runtime,
            population_stats=all_pop_stats,
            history=self.histories[0] if self.histories else None,
            success=success,
            termination_reason=termination_reason
        )
    
    def register_generation_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for generation events.
        
        Args:
            callback: Function to call after each generation
        """
        self.generation_callbacks.append(callback)
    
    def register_convergence_callback(self, callback: Callable[[EvolutionResult], None]) -> None:
        """Register a callback for convergence events.
        
        Args:
            callback: Function to call when evolution converges
        """
        self.convergence_callbacks.append(callback)
    
    def register_migration_callback(self, callback: Callable[[int, List[List[Individual]]], None]) -> None:
        """Register a callback for migration events.
        
        Args:
            callback: Function to call during migration
        """
        self.migration_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status of evolution."""
        return {
            'is_running': self.is_running,
            'current_generation': self.current_generation,
            'populations': len(self.populations),
            'mode': self.mode.value,
            'runtime': time.time() - (self.start_time or 0) if self.start_time else 0,
            'best_fitness': self.best_overall.fitness if self.best_overall else 0.0,
            'config': {
                'population_size': self.config.population_size,
                'max_generations': self.config.max_generations,
                'target_fitness': self.config.target_fitness
            }
        }
    
    def save_state(self, filepath: str) -> None:
        """Save orchestrator state to file.
        
        Args:
            filepath: Path to save state file
        """
        state = {
            'config': self.config.__dict__,
            'mode': self.mode.value,
            'current_generation': self.current_generation,
            'parameter_history': self.parameter_history,
            'best_individual': {
                'fitness': self.best_overall.fitness if self.best_overall else None,
                'genes': len(self.best_overall.genes) if self.best_overall else 0
            } if self.best_overall else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Orchestrator state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load orchestrator state from file.
        
        Args:
            filepath: Path to state file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore configuration
        config_dict = state.get('config', {})
        self.config = EvolutionConfig(**config_dict)
        
        # Restore other state
        self.mode = OrchestrationMode(state.get('mode', 'adaptive'))
        self.current_generation = state.get('current_generation', 0)
        self.parameter_history = state.get('parameter_history', [])
        
        logger.info(f"Orchestrator state loaded from {filepath}")


class CognitiveEvolutionCoordinator:
    """Coordinates evolution with cognitive modules in CogPrime architecture."""
    
    def __init__(self, atomspace: AtomSpace, orchestrator: EvolutionaryOrchestrator):
        """Initialize cognitive evolution coordinator.
        
        Args:
            atomspace: The AtomSpace for cognitive operations
            orchestrator: The evolutionary orchestrator
        """
        self.atomspace = atomspace
        self.orchestrator = orchestrator
        
        # Integration with cognitive modules
        self.perception_integration = False
        self.reasoning_integration = False
        self.action_integration = False
        
        # Evolution feedback mechanisms
        self.fitness_feedback: List[Callable] = []
        self.structure_feedback: List[Callable] = []
    
    def integrate_with_perception(self, perception_module) -> None:
        """Integrate evolution with perception module."""
        # This would connect evolutionary feedback to perception
        self.perception_integration = True
        logger.info("Evolution integrated with perception module")
    
    def integrate_with_reasoning(self, reasoning_module) -> None:
        """Integrate evolution with reasoning module.""" 
        # This would connect evolutionary patterns to reasoning
        self.reasoning_integration = True
        logger.info("Evolution integrated with reasoning module")
    
    def integrate_with_action(self, action_module) -> None:
        """Integrate evolution with action module."""
        # This would evolve action strategies
        self.action_integration = True
        logger.info("Evolution integrated with action module")
    
    def evolve_cognitive_structures(self, target_capability: str, 
                                  context: Dict[str, Any] = None) -> EvolutionResult:
        """Evolve cognitive structures for a specific capability.
        
        Args:
            target_capability: The cognitive capability to evolve
            context: Context information for evolution
            
        Returns:
            Result of the evolutionary process
        """
        logger.info(f"Evolving cognitive structures for: {target_capability}")
        
        # Setup evolution for specific cognitive capability
        self._setup_capability_evolution(target_capability, context)
        
        # Run evolution
        result = self.orchestrator.start_evolution(async_mode=False)
        
        # Integrate results back into cognitive architecture
        if result.success:
            self._integrate_evolved_structures(result.best_individual, target_capability)
        
        return result
    
    def _setup_capability_evolution(self, capability: str, context: Dict[str, Any]) -> None:
        """Setup evolution parameters for specific cognitive capability."""
        # This would configure evolution based on the target capability
        if capability == "pattern_recognition":
            # Configure for pattern recognition evolution
            pass
        elif capability == "decision_making":
            # Configure for decision making evolution
            pass
        elif capability == "learning_adaptation":
            # Configure for learning adaptation evolution
            pass
        
        logger.debug(f"Evolution setup for capability: {capability}")
    
    def _integrate_evolved_structures(self, individual: Individual, capability: str) -> None:
        """Integrate evolved structures back into cognitive architecture."""
        # Convert individual to AtomSpace structures
        evolved_atoms = individual.to_atomspace_structure()
        
        # Add to AtomSpace with appropriate truth values
        for atom in evolved_atoms:
            # Set high truth value for evolved structures
            atom.set_truth_value(TruthValue(strength=0.9, confidence=0.8))
            self.atomspace.add(atom)
        
        logger.info(f"Integrated {len(evolved_atoms)} evolved structures for {capability}")