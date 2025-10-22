"""
Phoenix MOSES - Civilizational Evolution Engine

This module implements the Phoenix MOSES system, which adapts the MOSES
evolutionary algorithm specifically for civilizational transformation
and consciousness evolution within the Cycle of Phoenix framework.
"""

import random
import math
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass

from ..atomspace import AtomSpace, Node, Link, TruthValue, AttentionValue  
from ..evolution import MosesEngine, Individual, Gene, FitnessFunction
from .civilizational_engine import TransformationPhase, TransformationMetrics

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessLevel:
    """Represents a level of consciousness evolution."""
    name: str
    level: int  # 0-based level
    integration_capacity: float  # 0.0 to 1.0
    complexity_threshold: float  # Minimum complexity to reach this level
    coherence_requirement: float  # Required coherence for stability
    transformation_potential: float  # Potential for further transformation


class CivilizationalFitness(FitnessFunction):
    """
    Fitness function that evaluates individuals based on their contribution
    to civilizational transformation and consciousness evolution.
    """
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize civilizational fitness function.
        
        Args:
            atomspace: The AtomSpace for context
        """
        self.atomspace = atomspace
        
        # Civilizational context
        self.current_transformation_phase = TransformationPhase.EMERGENCE
        self.transformation_metrics = TransformationMetrics()
        self.consciousness_levels = self._initialize_consciousness_levels()
        
        # Fitness weights for different aspects
        self.fitness_weights = {
            'consciousness_evolution': 0.3,
            'civilizational_contribution': 0.3,
            'integration_capacity': 0.2,
            'transformation_potential': 0.2
        }
    
    def _initialize_consciousness_levels(self) -> List[ConsciousnessLevel]:
        """Initialize consciousness evolution levels."""
        return [
            ConsciousnessLevel(
                name="Instinctual",
                level=0,
                integration_capacity=0.2,
                complexity_threshold=0.1,
                coherence_requirement=0.3,
                transformation_potential=0.8
            ),
            ConsciousnessLevel(
                name="Symbolic",
                level=1,
                integration_capacity=0.4,
                complexity_threshold=0.3,
                coherence_requirement=0.5,
                transformation_potential=0.7
            ),
            ConsciousnessLevel(
                name="Rational", 
                level=2,
                integration_capacity=0.6,
                complexity_threshold=0.5,
                coherence_requirement=0.7,
                transformation_potential=0.6
            ),
            ConsciousnessLevel(
                name="Integral",
                level=3,
                integration_capacity=0.8,
                complexity_threshold=0.7,
                coherence_requirement=0.8,
                transformation_potential=0.5
            ),
            ConsciousnessLevel(
                name="Trans-rational",
                level=4,
                integration_capacity=0.9,
                complexity_threshold=0.8,
                coherence_requirement=0.9,
                transformation_potential=0.4
            ),
            ConsciousnessLevel(
                name="Cosmic",
                level=5,
                integration_capacity=1.0,
                complexity_threshold=0.9,
                coherence_requirement=0.95,
                transformation_potential=0.2
            )
        ]
    
    def evaluate(self, individual: Individual, context: Dict[str, Any] = None) -> float:
        """Evaluate individual fitness for civilizational evolution.
        
        Args:
            individual: Individual to evaluate
            context: Optional context information
            
        Returns:
            Fitness score (0.0 to 1.0)
        """
        if not individual.genes:
            return 0.0
        
        # Calculate component fitness scores
        consciousness_score = self._evaluate_consciousness_evolution(individual)
        civilizational_score = self._evaluate_civilizational_contribution(individual)
        integration_score = self._evaluate_integration_capacity(individual)
        transformation_score = self._evaluate_transformation_potential(individual)
        
        # Weighted combination
        total_fitness = (
            self.fitness_weights['consciousness_evolution'] * consciousness_score +
            self.fitness_weights['civilizational_contribution'] * civilizational_score +
            self.fitness_weights['integration_capacity'] * integration_score +
            self.fitness_weights['transformation_potential'] * transformation_score
        )
        
        # Apply phase-specific adjustments
        phase_adjusted_fitness = self._apply_phase_adjustment(individual, total_fitness)
        
        return max(0.0, min(1.0, phase_adjusted_fitness))
    
    def _evaluate_consciousness_evolution(self, individual: Individual) -> float:
        """Evaluate individual's contribution to consciousness evolution.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Consciousness evolution score
        """
        active_genes = individual.get_active_genes()
        
        if not active_genes:
            return 0.0
        
        # Calculate consciousness indicators
        complexity = self._calculate_gene_complexity(active_genes)
        coherence = self._calculate_gene_coherence(active_genes)
        integration = self._calculate_gene_integration(active_genes)
        
        # Map to consciousness level
        consciousness_level = self._determine_consciousness_level(complexity, coherence, integration)
        
        # Score based on consciousness level and current phase needs
        base_score = (consciousness_level.level + 1) / len(self.consciousness_levels)
        
        # Bonus for meeting current transformation needs
        phase_bonus = 0.0
        if self.current_transformation_phase in [TransformationPhase.EMERGENCE, TransformationPhase.RENEWAL]:
            # Reward higher transformation potential in emergence/renewal phases
            phase_bonus = consciousness_level.transformation_potential * 0.2
        elif self.current_transformation_phase in [TransformationPhase.INTEGRATION, TransformationPhase.CRYSTALLIZATION]:
            # Reward higher integration capacity in stable phases
            phase_bonus = consciousness_level.integration_capacity * 0.2
        
        return base_score + phase_bonus
    
    def _calculate_gene_complexity(self, genes: List[Gene]) -> float:
        """Calculate complexity of gene structure.
        
        Args:
            genes: List of genes to analyze
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        if not genes:
            return 0.0
        
        # Complexity based on gene diversity and relationships
        atom_types = set(gene.atom.atom_type for gene in genes)
        type_diversity = len(atom_types) / max(len(genes), 1)
        
        # Weight distribution complexity
        weights = [gene.weight for gene in genes]
        weight_variance = self._calculate_variance(weights)
        
        # Link complexity (for genes with Link atoms)
        link_complexity = 0.0
        link_count = 0
        
        for gene in genes:
            if gene.atom.is_link():
                arity = gene.atom.get_arity()
                link_complexity += arity / 10.0  # Normalize by max expected arity
                link_count += 1
        
        if link_count > 0:
            link_complexity /= link_count
        
        # Combine complexity measures
        complexity = (type_diversity + weight_variance + link_complexity) / 3.0
        return min(1.0, complexity)
    
    def _calculate_gene_coherence(self, genes: List[Gene]) -> float:
        """Calculate coherence of gene structure.
        
        Args:
            genes: List of genes to analyze
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not genes:
            return 0.0
        
        # Coherence based on semantic relationships between atoms
        total_coherence = 0.0
        comparisons = 0
        
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene1, gene2 = genes[i], genes[j]
                
                # Calculate semantic similarity
                similarity = self._calculate_atom_similarity(gene1.atom, gene2.atom)
                
                # Weight by gene weights
                weighted_similarity = similarity * gene1.weight * gene2.weight
                total_coherence += weighted_similarity
                comparisons += 1
        
        if comparisons == 0:
            return 1.0  # Single gene is coherent
        
        return total_coherence / comparisons
    
    def _calculate_gene_integration(self, genes: List[Gene]) -> float:
        """Calculate integration level of genes.
        
        Args:
            genes: List of genes to analyze
            
        Returns:
            Integration score (0.0 to 1.0)
        """
        if not genes:
            return 0.0
        
        # Integration based on how well genes work together
        active_count = sum(1 for gene in genes if gene.active)
        
        if active_count == 0:
            return 0.0
        
        # Balance between having active genes and not too many
        activation_balance = active_count / len(genes)
        optimal_activation = 0.7  # 70% active is considered optimal
        activation_score = 1.0 - abs(activation_balance - optimal_activation)
        
        # Weight distribution score
        active_weights = [gene.weight for gene in genes if gene.active]
        weight_harmony = 1.0 - self._calculate_variance(active_weights) if active_weights else 0.0
        
        # Atom type integration
        active_types = set(gene.atom.atom_type for gene in genes if gene.active)
        type_integration = len(active_types) / max(active_count, 1)
        
        # Combine integration measures
        integration = (activation_score + weight_harmony + type_integration) / 3.0
        return integration
    
    def _determine_consciousness_level(self, complexity: float, coherence: float, 
                                     integration: float) -> ConsciousnessLevel:
        """Determine consciousness level based on metrics.
        
        Args:
            complexity: Complexity score
            coherence: Coherence score  
            integration: Integration score
            
        Returns:
            Corresponding consciousness level
        """
        # Find highest level that meets requirements
        for level in reversed(self.consciousness_levels):
            if (complexity >= level.complexity_threshold and
                coherence >= level.coherence_requirement and
                integration >= level.integration_capacity):
                return level
        
        # Default to lowest level
        return self.consciousness_levels[0]
    
    def _evaluate_civilizational_contribution(self, individual: Individual) -> float:
        """Evaluate individual's contribution to civilizational development.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Civilizational contribution score
        """
        active_genes = individual.get_active_genes()
        
        if not active_genes:
            return 0.0
        
        # Evaluate contribution based on transformation metrics alignment
        complexity_contribution = self._evaluate_complexity_contribution(active_genes)
        stability_contribution = self._evaluate_stability_contribution(active_genes)
        adaptability_contribution = self._evaluate_adaptability_contribution(active_genes)
        
        # Weight contributions based on current transformation needs
        contributions = {
            'complexity': complexity_contribution,
            'stability': stability_contribution,
            'adaptability': adaptability_contribution
        }
        
        # Phase-specific weighting
        phase_weights = self._get_phase_contribution_weights()
        
        total_contribution = sum(
            contributions[aspect] * phase_weights.get(aspect, 1.0) / 3.0
            for aspect in contributions
        )
        
        return total_contribution
    
    def _evaluate_complexity_contribution(self, genes: List[Gene]) -> float:
        """Evaluate contribution to civilizational complexity."""
        complexity = self._calculate_gene_complexity(genes)
        
        # Current civilizational complexity need
        complexity_need = 1.0 - self.transformation_metrics.complexity
        
        # Score based on how well individual addresses complexity need
        if complexity_need > 0.5:
            # Need more complexity
            return complexity
        else:
            # Need stability, not more complexity
            return 1.0 - complexity * 0.5
    
    def _evaluate_stability_contribution(self, genes: List[Gene]) -> float:
        """Evaluate contribution to civilizational stability."""
        coherence = self._calculate_gene_coherence(genes)
        integration = self._calculate_gene_integration(genes)
        
        stability_contribution = (coherence + integration) / 2.0
        
        # Current stability need
        stability_need = 1.0 - self.transformation_metrics.coherence
        
        # Score based on stability need
        if stability_need > 0.5:
            return stability_contribution
        else:
            # Too much stability can lead to stagnation
            return stability_contribution * 0.7
    
    def _evaluate_adaptability_contribution(self, genes: List[Gene]) -> float:
        """Evaluate contribution to civilizational adaptability."""
        # Adaptability comes from diversity and flexibility
        active_genes = [gene for gene in genes if gene.active]
        
        if not active_genes:
            return 0.0
        
        # Measure flexibility in gene activation
        activation_flexibility = len(active_genes) / len(genes)
        
        # Measure diversity
        atom_types = set(gene.atom.atom_type for gene in active_genes)
        type_diversity = len(atom_types) / len(active_genes)
        
        # Weight variance (higher variance = more adaptability)
        weights = [gene.weight for gene in active_genes]
        weight_variance = self._calculate_variance(weights)
        
        adaptability = (activation_flexibility + type_diversity + weight_variance) / 3.0
        
        return adaptability
    
    def _get_phase_contribution_weights(self) -> Dict[str, float]:
        """Get contribution weights based on current transformation phase."""
        phase_weights = {
            TransformationPhase.EMERGENCE: {
                'complexity': 1.2,
                'stability': 0.8,
                'adaptability': 1.3
            },
            TransformationPhase.INTEGRATION: {
                'complexity': 1.1,
                'stability': 1.3,
                'adaptability': 1.0
            },
            TransformationPhase.CRYSTALLIZATION: {
                'complexity': 0.9,
                'stability': 1.4,
                'adaptability': 0.7
            },
            TransformationPhase.STAGNATION: {
                'complexity': 0.8,
                'stability': 1.2,
                'adaptability': 1.1
            },
            TransformationPhase.BREAKDOWN: {
                'complexity': 0.7,
                'stability': 0.6,
                'adaptability': 1.4
            },
            TransformationPhase.COLLAPSE: {
                'complexity': 0.5,
                'stability': 0.4,
                'adaptability': 1.5
            },
            TransformationPhase.RENEWAL: {
                'complexity': 1.3,
                'stability': 0.9,
                'adaptability': 1.4
            }
        }
        
        return phase_weights.get(self.current_transformation_phase, 
                               {'complexity': 1.0, 'stability': 1.0, 'adaptability': 1.0})
    
    def _evaluate_integration_capacity(self, individual: Individual) -> float:
        """Evaluate individual's capacity for integration."""
        active_genes = individual.get_active_genes()
        
        if not active_genes:
            return 0.0
        
        return self._calculate_gene_integration(active_genes)
    
    def _evaluate_transformation_potential(self, individual: Individual) -> float:
        """Evaluate individual's potential for driving transformation."""
        active_genes = individual.get_active_genes()
        
        if not active_genes:
            return 0.0
        
        # Transformation potential based on novelty and adaptability
        complexity = self._calculate_gene_complexity(active_genes)
        adaptability = self._evaluate_adaptability_contribution(active_genes)
        
        # Balance between being complex enough to drive change
        # but not so complex as to be incoherent
        coherence = self._calculate_gene_coherence(active_genes)
        
        # Transformation potential formula
        transformation_potential = (complexity + adaptability) / 2.0
        
        # Apply coherence constraint
        if coherence < 0.3:
            transformation_potential *= coherence / 0.3  # Penalize low coherence
        
        return transformation_potential
    
    def _apply_phase_adjustment(self, individual: Individual, base_fitness: float) -> float:
        """Apply phase-specific fitness adjustments.
        
        Args:
            individual: Individual being evaluated
            base_fitness: Base fitness score
            
        Returns:
            Phase-adjusted fitness score
        """
        # Phase-specific multipliers
        phase_multipliers = {
            TransformationPhase.EMERGENCE: 1.1,      # Reward innovation
            TransformationPhase.INTEGRATION: 1.0,    # Balanced evaluation
            TransformationPhase.CRYSTALLIZATION: 0.9, # Slightly conservative
            TransformationPhase.STAGNATION: 1.2,     # Reward change agents
            TransformationPhase.BREAKDOWN: 1.3,      # Strongly reward adaptability
            TransformationPhase.COLLAPSE: 1.4,       # Desperate need for solutions
            TransformationPhase.RENEWAL: 1.2         # Reward renewal capacity
        }
        
        multiplier = phase_multipliers.get(self.current_transformation_phase, 1.0)
        
        # Apply civilization vitality modifier
        vitality = self.transformation_metrics.get_overall_vitality()
        vitality_modifier = 0.8 + 0.4 * vitality  # Range: 0.8 to 1.2
        
        adjusted_fitness = base_fitness * multiplier * vitality_modifier
        
        return adjusted_fitness
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        return variance
    
    def _calculate_atom_similarity(self, atom1, atom2) -> float:
        """Calculate similarity between two atoms."""
        # Simple similarity based on type and structure
        if atom1.atom_type == atom2.atom_type:
            if atom1.is_node() and atom2.is_node():
                return 0.8 if atom1.name == atom2.name else 0.5
            elif atom1.is_link() and atom2.is_link():
                if len(atom1.outgoing_set) == len(atom2.outgoing_set):
                    return 0.7
                else:
                    return 0.3
            return 0.5
        return 0.2
    
    def update_civilizational_context(self, context: Dict[str, Any]) -> None:
        """Update civilizational context for fitness evaluation.
        
        Args:
            context: Context information from civilizational engine
        """
        # Update transformation phase
        phase_str = context.get('phase', 'emergence')
        try:
            self.current_transformation_phase = TransformationPhase(phase_str)
        except ValueError:
            logger.warning(f"Unknown transformation phase: {phase_str}")
        
        # Update transformation metrics if provided
        if 'metrics' in context:
            metrics_dict = context['metrics']
            for attr, value in metrics_dict.items():
                if hasattr(self.transformation_metrics, attr):
                    setattr(self.transformation_metrics, attr, value)


class PhoenixMOSES:
    """
    Phoenix MOSES - Evolutionary system adapted for civilizational transformation.
    
    This class extends the standard MOSES engine with civilization-specific
    capabilities, including consciousness evolution and transformation dynamics.
    """
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize Phoenix MOSES system.
        
        Args:
            atomspace: The AtomSpace for operations
        """
        self.atomspace = atomspace
        
        # Create specialized fitness function
        self.civilizational_fitness = CivilizationalFitness(atomspace)
        
        # Create MOSES engine with civilizational fitness
        self.moses_engine = MosesEngine(atomspace, population_size=50)
        self.moses_engine.fitness_function = self.civilizational_fitness
        
        # Phoenix-specific parameters
        self.consciousness_evolution_target = 0.8
        self.civilizational_integration_threshold = 0.7
        
        # Evolution tracking
        self.evolution_history: List[Dict[str, Any]] = []
        self.consciousness_progression: List[float] = []
        
        logger.info("Phoenix MOSES initialized")
    
    def evolve_consciousness(self, target_level: int = 3, 
                           max_generations: int = 100) -> Dict[str, Any]:
        """Evolve toward a target consciousness level.
        
        Args:
            target_level: Target consciousness level (0-5)
            max_generations: Maximum generations to evolve
            
        Returns:
            Evolution results
        """
        logger.info(f"Evolving consciousness toward level {target_level}")
        
        # Adjust fitness weights for consciousness evolution
        self.civilizational_fitness.fitness_weights['consciousness_evolution'] = 0.5
        self.civilizational_fitness.fitness_weights['transformation_potential'] = 0.3
        
        # Initialize population if needed
        if not self.moses_engine.population:
            self.moses_engine.initialize_population()
        
        # Evolution loop
        best_consciousness_level = 0
        generations_without_improvement = 0
        stagnation_limit = 20
        
        for generation in range(max_generations):
            # Evolve one generation
            self.moses_engine.evolve_generation()
            
            # Evaluate consciousness progress
            best_individual = self.moses_engine.get_best_individual()
            
            if best_individual:
                current_level = self._assess_consciousness_level(best_individual)
                self.consciousness_progression.append(current_level)
                
                # Check for improvement
                if current_level > best_consciousness_level:
                    best_consciousness_level = current_level
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Check for target achievement
                if current_level >= target_level:
                    logger.info(f"Target consciousness level {target_level} achieved in {generation + 1} generations")
                    break
                
                # Check for stagnation
                if generations_without_improvement >= stagnation_limit:
                    logger.info(f"Evolution stagnated after {generation + 1} generations")
                    break
        
        # Generate results
        results = {
            'target_level': target_level,
            'achieved_level': best_consciousness_level,
            'generations': generation + 1,
            'best_individual': best_individual,
            'consciousness_progression': self.consciousness_progression.copy(),
            'final_population_stats': self.moses_engine.get_population_stats()
        }
        
        return results
    
    def evolve_for_civilizational_phase(self, phase: TransformationPhase,
                                      max_generations: int = 50) -> Dict[str, Any]:
        """Evolve solutions for a specific civilizational phase.
        
        Args:
            phase: Target transformation phase
            max_generations: Maximum generations to evolve
            
        Returns:
            Evolution results
        """
        logger.info(f"Evolving for civilizational phase: {phase.value}")
        
        # Update civilizational context
        self.civilizational_fitness.current_transformation_phase = phase
        
        # Adjust fitness weights based on phase
        phase_fitness_weights = self._get_phase_fitness_weights(phase)
        self.civilizational_fitness.fitness_weights.update(phase_fitness_weights)
        
        # Initialize population if needed
        if not self.moses_engine.population:
            self.moses_engine.initialize_population()
        
        # Evolution loop
        best_fitness_history = []
        
        for generation in range(max_generations):
            self.moses_engine.evolve_generation()
            
            # Track progress
            stats = self.moses_engine.get_population_stats()
            best_fitness = stats.get('best_fitness', 0.0)
            best_fitness_history.append(best_fitness)
            
            # Record evolution state
            if generation % 10 == 0:
                self._record_evolution_state(generation, phase)
        
        # Generate results
        best_individual = self.moses_engine.get_best_individual()
        
        results = {
            'phase': phase.value,
            'generations': max_generations,
            'best_individual': best_individual,
            'best_fitness': best_fitness_history[-1] if best_fitness_history else 0.0,
            'fitness_progression': best_fitness_history,
            'final_population_stats': self.moses_engine.get_population_stats(),
            'civilizational_contribution': self._assess_civilizational_contribution(best_individual)
        }
        
        return results
    
    def _get_phase_fitness_weights(self, phase: TransformationPhase) -> Dict[str, float]:
        """Get fitness weights optimized for a specific phase.
        
        Args:
            phase: Transformation phase
            
        Returns:
            Dictionary of fitness weights
        """
        phase_weights = {
            TransformationPhase.EMERGENCE: {
                'consciousness_evolution': 0.4,
                'civilizational_contribution': 0.2,
                'integration_capacity': 0.1,
                'transformation_potential': 0.3
            },
            TransformationPhase.INTEGRATION: {
                'consciousness_evolution': 0.3,
                'civilizational_contribution': 0.3,
                'integration_capacity': 0.3,
                'transformation_potential': 0.1
            },
            TransformationPhase.CRYSTALLIZATION: {
                'consciousness_evolution': 0.2,
                'civilizational_contribution': 0.4,
                'integration_capacity': 0.3,
                'transformation_potential': 0.1
            },
            TransformationPhase.STAGNATION: {
                'consciousness_evolution': 0.3,
                'civilizational_contribution': 0.2,
                'integration_capacity': 0.2,
                'transformation_potential': 0.3
            },
            TransformationPhase.BREAKDOWN: {
                'consciousness_evolution': 0.4,
                'civilizational_contribution': 0.1,
                'integration_capacity': 0.2,
                'transformation_potential': 0.3
            },
            TransformationPhase.COLLAPSE: {
                'consciousness_evolution': 0.5,
                'civilizational_contribution': 0.1,
                'integration_capacity': 0.1,
                'transformation_potential': 0.3
            },
            TransformationPhase.RENEWAL: {
                'consciousness_evolution': 0.4,
                'civilizational_contribution': 0.2,
                'integration_capacity': 0.2,
                'transformation_potential': 0.2
            }
        }
        
        return phase_weights.get(phase, {
            'consciousness_evolution': 0.25,
            'civilizational_contribution': 0.25,
            'integration_capacity': 0.25,
            'transformation_potential': 0.25
        })
    
    def _assess_consciousness_level(self, individual: Individual) -> float:
        """Assess consciousness level of an individual.
        
        Args:
            individual: Individual to assess
            
        Returns:
            Consciousness level (0.0 to 5.0)
        """
        if not individual or not individual.genes:
            return 0.0
        
        active_genes = individual.get_active_genes()
        
        # Calculate consciousness metrics
        complexity = self.civilizational_fitness._calculate_gene_complexity(active_genes)
        coherence = self.civilizational_fitness._calculate_gene_coherence(active_genes)
        integration = self.civilizational_fitness._calculate_gene_integration(active_genes)
        
        # Determine consciousness level
        consciousness_level = self.civilizational_fitness._determine_consciousness_level(
            complexity, coherence, integration
        )
        
        return float(consciousness_level.level)
    
    def _assess_civilizational_contribution(self, individual: Individual) -> Dict[str, float]:
        """Assess individual's civilizational contribution.
        
        Args:
            individual: Individual to assess
            
        Returns:
            Dictionary of contribution metrics
        """
        if not individual:
            return {}
        
        active_genes = individual.get_active_genes()
        
        return {
            'complexity_contribution': self.civilizational_fitness._evaluate_complexity_contribution(active_genes),
            'stability_contribution': self.civilizational_fitness._evaluate_stability_contribution(active_genes),
            'adaptability_contribution': self.civilizational_fitness._evaluate_adaptability_contribution(active_genes),
            'overall_contribution': self.civilizational_fitness._evaluate_civilizational_contribution(individual)
        }
    
    def _record_evolution_state(self, generation: int, phase: TransformationPhase) -> None:
        """Record current evolution state.
        
        Args:
            generation: Current generation number
            phase: Current transformation phase
        """
        best_individual = self.moses_engine.get_best_individual()
        
        state_record = {
            'generation': generation,
            'phase': phase.value,
            'population_stats': self.moses_engine.get_population_stats(),
            'consciousness_level': self._assess_consciousness_level(best_individual) if best_individual else 0.0,
            'civilizational_contribution': self._assess_civilizational_contribution(best_individual)
        }
        
        self.evolution_history.append(state_record)
        
        # Limit history size
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-500:]
    
    def update_civilizational_context(self, context: Dict[str, Any]) -> None:
        """Update civilizational context for evolution.
        
        Args:
            context: Context information
        """
        self.civilizational_fitness.update_civilizational_context(context)
    
    def get_evolution_insights(self) -> Dict[str, Any]:
        """Get insights from evolution history.
        
        Returns:
            Dictionary of evolution insights
        """
        if not self.evolution_history:
            return {'error': 'No evolution history available'}
        
        # Analyze consciousness progression
        consciousness_levels = [record.get('consciousness_level', 0.0) 
                              for record in self.evolution_history]
        
        # Analyze fitness progression
        fitness_values = [record.get('population_stats', {}).get('best_fitness', 0.0)
                         for record in self.evolution_history]
        
        insights = {
            'total_generations': len(self.evolution_history),
            'consciousness_progression': {
                'initial_level': consciousness_levels[0] if consciousness_levels else 0.0,
                'final_level': consciousness_levels[-1] if consciousness_levels else 0.0,
                'max_level_achieved': max(consciousness_levels) if consciousness_levels else 0.0,
                'average_level': sum(consciousness_levels) / len(consciousness_levels) if consciousness_levels else 0.0
            },
            'fitness_progression': {
                'initial_fitness': fitness_values[0] if fitness_values else 0.0,
                'final_fitness': fitness_values[-1] if fitness_values else 0.0,
                'max_fitness': max(fitness_values) if fitness_values else 0.0,
                'fitness_trend': 'improving' if (fitness_values and fitness_values[-1] > fitness_values[0]) else 'declining'
            },
            'phase_performance': self._analyze_phase_performance()
        }
        
        return insights
    
    def _analyze_phase_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different transformation phases.
        
        Returns:
            Phase performance analysis
        """
        phase_performance = {}
        
        # Group records by phase
        records_by_phase = {}
        for record in self.evolution_history:
            phase = record.get('phase', 'unknown')
            if phase not in records_by_phase:
                records_by_phase[phase] = []
            records_by_phase[phase].append(record)
        
        # Analyze each phase
        for phase, records in records_by_phase.items():
            fitness_values = [record.get('population_stats', {}).get('best_fitness', 0.0) 
                            for record in records]
            consciousness_values = [record.get('consciousness_level', 0.0) 
                                  for record in records]
            
            if fitness_values and consciousness_values:
                phase_performance[phase] = {
                    'average_fitness': sum(fitness_values) / len(fitness_values),
                    'max_fitness': max(fitness_values),
                    'average_consciousness': sum(consciousness_values) / len(consciousness_values),
                    'max_consciousness': max(consciousness_values),
                    'generation_count': len(records)
                }
        
        return phase_performance