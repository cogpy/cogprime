"""
Integration Modules for Antikythera Framework

This module provides integration capabilities between the Antikythera orchestration
framework and other CogPrime components, including cognitive modules and evolutionary
systems.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.atomspace import AtomSpace, Node, Link, TruthValue, AttentionValue
from src.evolution import EvolutionaryOrchestrator, MosesEngine, Individual
from .cycles import CivilizationalCycle, MetaCycle, PhaseType

# Configure logging
logger = logging.getLogger(__name__)


class IntegrationModule(ABC):
    """Abstract base class for integration modules."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the integration module."""
        pass
    
    @abstractmethod
    def update(self, context: Dict[str, Any]) -> None:
        """Update the integration module with context."""
        pass
    
    @abstractmethod
    def get_feedback(self) -> Dict[str, Any]:
        """Get feedback from the integration module."""
        pass


class CognitiveIntegration(IntegrationModule):
    """Integration between Antikythera cycles and cognitive systems."""
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize cognitive integration.
        
        Args:
            atomspace: The AtomSpace for cognitive operations
        """
        self.atomspace = atomspace
        self.cycle_nodes: Dict[str, Node] = {}
        self.pattern_nodes: Dict[str, Node] = {}
        
        # Cognitive mappings
        self.phase_mappings: Dict[PhaseType, Dict[str, Any]] = {}
        self.attention_modulation: Dict[str, float] = {}
        
        # Feedback mechanisms
        self.cognitive_feedback: List[Callable] = []
        
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize cognitive integration."""
        # Create base cognitive structures for cycle integration
        self._create_cycle_ontology()
        self._setup_phase_mappings()
        
        logger.info("Cognitive integration initialized")
    
    def _create_cycle_ontology(self) -> None:
        """Create ontological structures for civilizational cycles in AtomSpace."""
        # Create root concept for civilizational cycles
        cycle_root = Node("ConceptNode", "CivilizationalCycle")
        self.atomspace.add(cycle_root)
        
        # Create phase type nodes
        phase_types = ["rise", "peak", "fall", "transition"]
        for phase_type in phase_types:
            phase_node = Node("ConceptNode", f"CyclePhase_{phase_type}")
            
            # Link to root concept
            inheritance_link = Link("InheritanceLink", [phase_node, cycle_root])
            
            self.atomspace.add(phase_node)
            self.atomspace.add(inheritance_link)
            self.pattern_nodes[f"phase_{phase_type}"] = phase_node
        
        # Create characteristic nodes
        characteristics = ["energy", "stability", "innovation_rate", 
                         "integration_level", "transformation_intensity"]
        
        for char in characteristics:
            char_node = Node("PredicateNode", f"has_{char}")
            self.atomspace.add(char_node)
            self.pattern_nodes[f"characteristic_{char}"] = char_node
    
    def _setup_phase_mappings(self) -> None:
        """Setup mappings between cycle phases and cognitive processes."""
        self.phase_mappings = {
            PhaseType.RISE: {
                'attention_focus': 0.7,      # Moderate focus on new patterns
                'learning_rate': 0.8,        # High learning rate
                'exploration_bias': 0.8,     # High exploration
                'memory_consolidation': 0.5, # Moderate consolidation
                'pattern_recognition': 0.6   # Building new patterns
            },
            PhaseType.PEAK: {
                'attention_focus': 0.9,      # High focused attention
                'learning_rate': 0.4,        # Lower learning rate
                'exploration_bias': 0.3,     # Low exploration (exploitation)
                'memory_consolidation': 0.9, # High consolidation
                'pattern_recognition': 0.9   # Strong pattern recognition
            },
            PhaseType.FALL: {
                'attention_focus': 0.5,      # Scattered attention
                'learning_rate': 0.9,        # Very high learning rate
                'exploration_bias': 0.9,     # High exploration of new patterns
                'memory_consolidation': 0.3, # Low consolidation
                'pattern_recognition': 0.4   # Breakdown of old patterns
            },
            PhaseType.TRANSITION: {
                'attention_focus': 0.6,      # Moderate attention
                'learning_rate': 0.7,        # High learning
                'exploration_bias': 0.7,     # High exploration
                'memory_consolidation': 0.6, # Moderate consolidation
                'pattern_recognition': 0.5   # Pattern reconstruction
            }
        }
    
    def update(self, context: Dict[str, Any]) -> None:
        """Update cognitive integration with cycle context.
        
        Args:
            context: Context from civilizational cycles
        """
        # Extract cycle characteristics
        meta_characteristics = context.get('meta_characteristics', {})
        active_cycles = context.get('active_cycles', [])
        
        # Update AtomSpace structures
        self._update_cycle_representations(active_cycles)
        
        # Modulate attention based on cycle state
        self._modulate_attention_economy(meta_characteristics)
        
        # Update cognitive parameters
        self._update_cognitive_parameters(meta_characteristics)
    
    def _update_cycle_representations(self, active_cycles: List[Dict[str, Any]]) -> None:
        """Update AtomSpace representations of active cycles.
        
        Args:
            active_cycles: List of active cycle information
        """
        for cycle_info in active_cycles:
            cycle_name = cycle_info.get('name', 'unknown')
            characteristics = cycle_info.get('characteristics', {})
            
            # Get or create cycle node
            cycle_node_name = f"cycle_{cycle_name.replace(' ', '_')}"
            
            if cycle_node_name not in self.cycle_nodes:
                cycle_node = Node("ConceptNode", cycle_node_name)
                self.atomspace.add(cycle_node)
                self.cycle_nodes[cycle_node_name] = cycle_node
            else:
                cycle_node = self.cycle_nodes[cycle_node_name]
            
            # Update truth values based on characteristics
            energy = characteristics.get('energy', 0.5)
            stability = characteristics.get('stability', 0.5)
            
            # Set truth value: strength from energy, confidence from stability
            tv = TruthValue(strength=energy, confidence=stability)
            cycle_node.set_truth_value(tv)
            
            # Update attention value
            sti = (energy - 0.5) * 100  # Short-term importance
            lti = stability * 50         # Long-term importance
            av = AttentionValue(sti=sti, lti=lti)
            cycle_node.set_attention_value(av)
            
            # Create links to characteristic nodes
            for char_name, char_value in characteristics.items():
                if f"characteristic_{char_name}" in self.pattern_nodes:
                    char_node = self.pattern_nodes[f"characteristic_{char_name}"]
                    
                    # Create evaluation link
                    eval_link = Link("EvaluationLink", [
                        char_node,
                        Link("ListLink", [cycle_node])
                    ])
                    
                    # Set truth value based on characteristic strength
                    eval_tv = TruthValue(strength=char_value, confidence=0.9)
                    eval_link.set_truth_value(eval_tv)
                    
                    self.atomspace.add(eval_link)
    
    def _modulate_attention_economy(self, meta_characteristics: Dict[str, float]) -> None:
        """Modulate AtomSpace attention economy based on cycle characteristics.
        
        Args:
            meta_characteristics: Aggregate characteristics from all cycles
        """
        # Get current attention parameters
        energy = meta_characteristics.get('energy', 0.5)
        stability = meta_characteristics.get('stability', 0.5)
        transformation_intensity = meta_characteristics.get('transformation_intensity', 0.5)
        
        # Calculate attention modulation parameters
        attention_spread = 1.0 - stability  # Less stable = more spread attention
        attention_intensity = energy        # More energy = higher attention
        attention_volatility = transformation_intensity  # More transformation = more volatility
        
        # Update attention values for all atoms in AtomSpace
        all_atoms = self.atomspace.get_all_atoms()
        
        for atom in all_atoms:
            current_av = atom.get_attention_value()
            
            # Apply modulation
            new_sti = current_av.sti * attention_intensity
            new_sti += (attention_volatility - 0.5) * 10  # Add volatility effect
            
            new_lti = current_av.lti * (1.0 - attention_spread)
            
            # Apply bounds
            new_sti = max(-100, min(100, new_sti))
            new_lti = max(0, min(100, new_lti))
            
            # Update attention value
            new_av = AttentionValue(sti=new_sti, lti=new_lti, vlti=current_av.vlti)
            atom.set_attention_value(new_av)
        
        self.attention_modulation = {
            'intensity': attention_intensity,
            'spread': attention_spread,
            'volatility': attention_volatility
        }
    
    def _update_cognitive_parameters(self, meta_characteristics: Dict[str, float]) -> None:
        """Update cognitive parameters based on cycle characteristics.
        
        Args:
            meta_characteristics: Aggregate characteristics from all cycles
        """
        # This would integrate with cognitive modules to update their parameters
        # For now, we'll store the parameters for potential use
        
        cognitive_params = {
            'learning_rate_multiplier': meta_characteristics.get('energy', 0.5) * 2.0,
            'attention_focus_factor': meta_characteristics.get('stability', 0.5),
            'exploration_bias': 1.0 - meta_characteristics.get('stability', 0.5),
            'memory_consolidation_rate': meta_characteristics.get('integration_level', 0.5),
            'pattern_sensitivity': meta_characteristics.get('transformation_intensity', 0.5)
        }
        
        # Store parameters that can be accessed by cognitive modules
        self.atomspace.metadata = getattr(self.atomspace, 'metadata', {})
        self.atomspace.metadata['cognitive_parameters'] = cognitive_params
    
    def get_feedback(self) -> Dict[str, Any]:
        """Get cognitive feedback for cycle adjustment.
        
        Returns:
            Dictionary containing cognitive feedback
        """
        feedback = {
            'attention_modulation': self.attention_modulation,
            'active_cycle_nodes': len(self.cycle_nodes),
            'attention_economy_health': self._assess_attention_economy(),
            'cognitive_coherence': self._measure_cognitive_coherence()
        }
        
        return feedback
    
    def _assess_attention_economy(self) -> Dict[str, float]:
        """Assess the health of the attention economy.
        
        Returns:
            Assessment metrics for attention economy
        """
        all_atoms = self.atomspace.get_all_atoms()
        
        if not all_atoms:
            return {'health_score': 0.0}
        
        # Calculate attention distribution metrics
        sti_values = [atom.get_attention_value().sti for atom in all_atoms]
        lti_values = [atom.get_attention_value().lti for atom in all_atoms]
        
        # Calculate statistics
        avg_sti = sum(sti_values) / len(sti_values)
        avg_lti = sum(lti_values) / len(lti_values)
        
        sti_variance = sum((x - avg_sti) ** 2 for x in sti_values) / len(sti_values)
        lti_variance = sum((x - avg_lti) ** 2 for x in lti_values) / len(lti_values)
        
        # Health metrics
        attention_balance = 1.0 - abs(avg_sti) / 100.0  # Closer to 0 is better
        attention_diversity = min(1.0, (sti_variance + lti_variance) / 5000.0)
        
        health_score = (attention_balance + attention_diversity) / 2.0
        
        return {
            'health_score': health_score,
            'attention_balance': attention_balance,
            'attention_diversity': attention_diversity,
            'avg_sti': avg_sti,
            'avg_lti': avg_lti
        }
    
    def _measure_cognitive_coherence(self) -> float:
        """Measure cognitive coherence across cycle representations.
        
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if len(self.cycle_nodes) < 2:
            return 1.0  # Perfect coherence with single or no cycles
        
        # Calculate coherence based on truth value similarities
        truth_values = []
        for cycle_node in self.cycle_nodes.values():
            tv = cycle_node.get_truth_value()
            truth_values.append((tv.strength, tv.confidence))
        
        if not truth_values:
            return 0.0
        
        # Calculate variance in truth values
        avg_strength = sum(tv[0] for tv in truth_values) / len(truth_values)
        avg_confidence = sum(tv[1] for tv in truth_values) / len(truth_values)
        
        strength_variance = sum((tv[0] - avg_strength) ** 2 for tv in truth_values) / len(truth_values)
        confidence_variance = sum((tv[1] - avg_confidence) ** 2 for tv in truth_values) / len(truth_values)
        
        # Coherence is inverse of variance
        total_variance = strength_variance + confidence_variance
        coherence = 1.0 / (1.0 + total_variance * 10.0)  # Scale factor for normalization
        
        return min(1.0, max(0.0, coherence))
    
    def register_cognitive_feedback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for cognitive feedback.
        
        Args:
            callback: Function to call with cognitive feedback
        """
        self.cognitive_feedback.append(callback)
    
    def update_from_cycles(self, meta_characteristics: Dict[str, float]) -> None:
        """Update cognitive integration from cycle characteristics.
        
        Args:
            meta_characteristics: Characteristics from meta-cycle
        """
        context = {'meta_characteristics': meta_characteristics}
        self.update(context)


class EvolutionaryIntegration(IntegrationModule):
    """Integration between Antikythera cycles and evolutionary systems."""
    
    def __init__(self, evolutionary_orchestrator: EvolutionaryOrchestrator, 
                 atomspace: AtomSpace):
        """Initialize evolutionary integration.
        
        Args:
            evolutionary_orchestrator: The evolutionary orchestrator to integrate
            atomspace: The AtomSpace for operations
        """
        self.evolutionary_orchestrator = evolutionary_orchestrator
        self.atomspace = atomspace
        
        # Integration state
        self.is_active = False
        self.cycle_fitness_functions: Dict[str, Callable] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Evolution parameters influenced by cycles
        self.cycle_evolution_parameters: Dict[str, Dict[str, float]] = {}
        
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize evolutionary integration."""
        # Setup cycle-aware fitness functions
        self._setup_cycle_fitness_functions()
        
        # Initialize evolution parameter mappings
        self._initialize_parameter_mappings()
        
        logger.info("Evolutionary integration initialized")
    
    def _setup_cycle_fitness_functions(self) -> None:
        """Setup fitness functions that are aware of civilizational cycles."""
        
        def cycle_aware_fitness(individual: Individual, cycle_characteristics: Dict[str, float]) -> float:
            """Fitness function that considers civilizational cycle state.
            
            Args:
                individual: Individual to evaluate
                cycle_characteristics: Current cycle characteristics
                
            Returns:
                Fitness score adjusted for cycle state
            """
            # Base fitness from individual
            base_fitness = individual.fitness or 0.0
            
            # Get cycle-based adjustments
            energy = cycle_characteristics.get('energy', 0.5)
            stability = cycle_characteristics.get('stability', 0.5)
            transformation_intensity = cycle_characteristics.get('transformation_intensity', 0.5)
            
            # Adjust fitness based on cycle phase
            # High energy phases favor high-energy individuals
            energy_bonus = (energy - 0.5) * 0.2 if base_fitness > 0.7 else 0.0
            
            # High stability phases favor stable (consistent) individuals
            stability_bonus = stability * 0.1 if len(individual.genes) > 2 else 0.0
            
            # High transformation phases favor innovative individuals
            innovation_bonus = transformation_intensity * 0.15 if base_fitness > 0.6 else 0.0
            
            adjusted_fitness = base_fitness + energy_bonus + stability_bonus + innovation_bonus
            
            return max(0.0, min(1.0, adjusted_fitness))
        
        self.cycle_fitness_functions['cycle_aware'] = cycle_aware_fitness
    
    def _initialize_parameter_mappings(self) -> None:
        """Initialize mappings from cycle characteristics to evolution parameters."""
        self.cycle_evolution_parameters = {
            'rise_phase': {
                'mutation_rate': 0.15,      # Moderate mutation
                'crossover_rate': 0.8,      # High crossover
                'selection_pressure': 1.2,   # Moderate selection
                'population_growth': 1.1     # Slight growth
            },
            'peak_phase': {
                'mutation_rate': 0.05,      # Low mutation
                'crossover_rate': 0.6,      # Moderate crossover
                'selection_pressure': 2.0,   # High selection pressure
                'population_growth': 1.0     # Stable population
            },
            'fall_phase': {
                'mutation_rate': 0.25,      # High mutation
                'crossover_rate': 0.9,      # Very high crossover
                'selection_pressure': 0.8,   # Low selection pressure
                'population_growth': 0.9     # Population decline
            },
            'transition_phase': {
                'mutation_rate': 0.2,       # High mutation
                'crossover_rate': 0.7,      # Moderate-high crossover
                'selection_pressure': 1.0,   # Neutral selection
                'population_growth': 1.05    # Slight growth
            }
        }
    
    def start_integration(self) -> None:
        """Start evolutionary integration."""
        self.is_active = True
        logger.info("Started evolutionary integration with cycles")
    
    def stop_integration(self) -> None:
        """Stop evolutionary integration."""
        self.is_active = False
        logger.info("Stopped evolutionary integration")
    
    def update(self, context: Dict[str, Any]) -> None:
        """Update evolutionary integration with context.
        
        Args:
            context: Context information
        """
        if not self.is_active:
            return
        
        # Extract cycle characteristics
        meta_characteristics = context.get('meta_characteristics', {})
        
        # Adapt evolution parameters
        self.adapt_evolution_parameters(meta_characteristics)
        
        # Record adaptation
        adaptation_record = {
            'timestamp': context.get('timestamp', 0),
            'characteristics': meta_characteristics,
            'adapted_parameters': self._get_current_evolution_parameters()
        }
        self.adaptation_history.append(adaptation_record)
        
        # Limit history size
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-500:]
    
    def adapt_evolution_parameters(self, cycle_characteristics: Dict[str, float]) -> None:
        """Adapt evolutionary parameters based on cycle characteristics.
        
        Args:
            cycle_characteristics: Current cycle characteristics
        """
        if not hasattr(self.evolutionary_orchestrator, 'engines') or not self.evolutionary_orchestrator.engines:
            return
        
        # Determine dominant phase type
        energy = cycle_characteristics.get('energy', 0.5)
        stability = cycle_characteristics.get('stability', 0.5)
        transformation_intensity = cycle_characteristics.get('transformation_intensity', 0.5)
        
        # Simple phase classification
        if energy > 0.6 and stability < 0.6:
            phase_type = 'rise_phase'
        elif energy > 0.7 and stability > 0.7:
            phase_type = 'peak_phase'
        elif stability < 0.5:
            phase_type = 'fall_phase'
        else:
            phase_type = 'transition_phase'
        
        # Get parameters for this phase
        if phase_type in self.cycle_evolution_parameters:
            target_params = self.cycle_evolution_parameters[phase_type]
            
            # Apply parameters to all evolution engines
            for engine in self.evolutionary_orchestrator.engines:
                self._apply_parameters_to_engine(engine, target_params, cycle_characteristics)
    
    def _apply_parameters_to_engine(self, engine: MosesEngine, 
                                  target_params: Dict[str, float],
                                  cycle_characteristics: Dict[str, float]) -> None:
        """Apply parameters to a specific evolution engine.
        
        Args:
            engine: The MOSES engine to modify
            target_params: Target parameters to apply
            cycle_characteristics: Current cycle characteristics for fine-tuning
        """
        # Update mutation rate
        if hasattr(engine, 'operators'):
            for operator in engine.operators:
                if hasattr(operator, 'mutation_rate'):
                    base_rate = target_params.get('mutation_rate', 0.1)
                    # Fine-tune based on transformation intensity
                    transformation_bonus = cycle_characteristics.get('transformation_intensity', 0.5) * 0.1
                    operator.mutation_rate = min(0.5, base_rate + transformation_bonus)
                
                if hasattr(operator, 'crossover_rate'):
                    base_rate = target_params.get('crossover_rate', 0.7)
                    # Fine-tune based on integration level
                    integration_factor = cycle_characteristics.get('integration_level', 0.5)
                    operator.crossover_rate = base_rate * (0.5 + 0.5 * integration_factor)
        
        # Update selection pressure
        if hasattr(engine, 'selection_pressure'):
            base_pressure = target_params.get('selection_pressure', 1.0)
            # Fine-tune based on stability
            stability_factor = cycle_characteristics.get('stability', 0.5)
            engine.selection_pressure = base_pressure * (0.5 + stability_factor)
    
    def sync_with_cycles(self, meta_cycle: MetaCycle) -> None:
        """Synchronize evolutionary progress with cycle dynamics.
        
        Args:
            meta_cycle: The meta-cycle to synchronize with
        """
        if not self.is_active:
            return
        
        # Get aggregate cycle state
        active_cycles = [cycle for cycle in meta_cycle.cycles if cycle.is_active]
        
        if not active_cycles:
            return
        
        # Calculate synchronization metrics
        total_progress = sum(cycle.get_cycle_progress() for cycle in active_cycles)
        avg_progress = total_progress / len(active_cycles)
        
        # Adjust evolutionary population based on cycle progress
        for engine in self.evolutionary_orchestrator.engines:
            # Influence population size based on cycle energy
            if hasattr(engine, 'population'):
                current_size = len(engine.population)
                
                # Get meta characteristics
                meta_chars = meta_cycle.get_meta_characteristics()
                energy = meta_chars.get('energy', 0.5)
                
                # Calculate target population size
                base_size = engine.population_size
                energy_factor = 0.8 + 0.4 * energy  # Range: 0.8 to 1.2
                target_size = int(base_size * energy_factor)
                
                # Gradually adjust population size
                if current_size < target_size:
                    # Add individuals (immigration)
                    self._add_immigrants_to_engine(engine, target_size - current_size)
                elif current_size > target_size:
                    # Remove individuals (selection pressure)
                    self._reduce_population_size(engine, current_size - target_size)
    
    def _add_immigrants_to_engine(self, engine: MosesEngine, count: int) -> None:
        """Add immigrant individuals to an engine population.
        
        Args:
            engine: The MOSES engine
            count: Number of immigrants to add
        """
        # Create new random individuals
        all_atoms = self.atomspace.get_all_atoms()
        
        if not all_atoms:
            return
        
        for _ in range(count):
            # Create random individual
            individual = Individual()
            
            # Add random genes
            num_genes = min(3, len(all_atoms))
            selected_atoms = [all_atoms[i % len(all_atoms)] for i in range(num_genes)]
            
            for atom in selected_atoms:
                from src.evolution.moses_engine import Gene
                gene = Gene(atom=atom, weight=0.5, active=True)
                individual.add_gene(gene)
            
            engine.population.append(individual)
    
    def _reduce_population_size(self, engine: MosesEngine, count: int) -> None:
        """Reduce population size by removing weakest individuals.
        
        Args:
            engine: The MOSES engine
            count: Number of individuals to remove
        """
        if len(engine.population) <= count:
            return
        
        # Evaluate population if needed
        engine.evaluate_population()
        
        # Sort by fitness (ascending - worst first)
        engine.population.sort(key=lambda ind: ind.fitness or 0.0)
        
        # Remove weakest individuals
        engine.population = engine.population[count:]
    
    def get_primary_engine(self) -> Optional[MosesEngine]:
        """Get the primary MOSES engine.
        
        Returns:
            The primary MOSES engine or None
        """
        if (hasattr(self.evolutionary_orchestrator, 'engines') and 
            self.evolutionary_orchestrator.engines):
            return self.evolutionary_orchestrator.engines[0]
        return None
    
    def _get_current_evolution_parameters(self) -> Dict[str, Any]:
        """Get current evolution parameters from all engines.
        
        Returns:
            Dictionary of current parameters
        """
        params = {
            'engines': len(self.evolutionary_orchestrator.engines) if hasattr(self.evolutionary_orchestrator, 'engines') else 0,
            'total_population': 0,
            'avg_mutation_rate': 0.0,
            'avg_selection_pressure': 0.0
        }
        
        if not hasattr(self.evolutionary_orchestrator, 'engines') or not self.evolutionary_orchestrator.engines:
            return params
        
        total_mutation = 0.0
        total_pressure = 0.0
        mutation_count = 0
        
        for engine in self.evolutionary_orchestrator.engines:
            params['total_population'] += len(engine.population) if hasattr(engine, 'population') else 0
            
            if hasattr(engine, 'operators'):
                for operator in engine.operators:
                    if hasattr(operator, 'mutation_rate'):
                        total_mutation += operator.mutation_rate
                        mutation_count += 1
            
            if hasattr(engine, 'selection_pressure'):
                total_pressure += engine.selection_pressure
        
        if mutation_count > 0:
            params['avg_mutation_rate'] = total_mutation / mutation_count
        
        if len(self.evolutionary_orchestrator.engines) > 0:
            params['avg_selection_pressure'] = total_pressure / len(self.evolutionary_orchestrator.engines)
        
        return params
    
    def get_feedback(self) -> Dict[str, Any]:
        """Get evolutionary feedback for cycle adjustment.
        
        Returns:
            Dictionary containing evolutionary feedback
        """
        feedback = {
            'is_active': self.is_active,
            'adaptation_history_length': len(self.adaptation_history),
            'current_parameters': self._get_current_evolution_parameters(),
            'fitness_functions': list(self.cycle_fitness_functions.keys())
        }
        
        # Add recent adaptation trend if available
        if len(self.adaptation_history) >= 2:
            recent = self.adaptation_history[-1]
            previous = self.adaptation_history[-2]
            
            feedback['recent_adaptation'] = {
                'timestamp_diff': recent['timestamp'] - previous['timestamp'],
                'parameter_changes': self._calculate_parameter_changes(previous, recent)
            }
        
        return feedback
    
    def _calculate_parameter_changes(self, previous: Dict[str, Any], 
                                   recent: Dict[str, Any]) -> Dict[str, float]:
        """Calculate changes in parameters between two adaptation records.
        
        Args:
            previous: Previous adaptation record
            recent: Recent adaptation record
            
        Returns:
            Dictionary of parameter changes
        """
        changes = {}
        
        prev_params = previous.get('adapted_parameters', {})
        recent_params = recent.get('adapted_parameters', {})
        
        for key in set(prev_params.keys()) | set(recent_params.keys()):
            if key in prev_params and key in recent_params:
                if isinstance(prev_params[key], (int, float)) and isinstance(recent_params[key], (int, float)):
                    changes[key] = recent_params[key] - prev_params[key]
        
        return changes