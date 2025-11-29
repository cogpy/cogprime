"""
Ennead-Based Relevance Realization Optimizer

This module implements optimal relevance realization through integration
of ten dimensions organized in a triad-of-triads structure (3+3+3 core + 1 transformative):

Triad I - Ways of Knowing (Epistemological): 4 dimensions
  1. Propositional (knowing-that)
  2. Procedural (knowing-how)  
  3. Perspectival (knowing-as)
  4. Participatory (knowing-by-being) - The transformative dimension

Triad II - Orders of Understanding (Ontological): 3 dimensions
  5. Nomological Order (How things work)
  6. Normative Order (What matters)
  7. Narrative Order (How things develop)

Triad III - Practices of Wisdom (Axiological): 3 dimensions
  8. Morality (Virtue & character)
  9. Meaning (Coherence & purpose)
  10. Mastery (Excellence & flow)

Note: While called "Ennead" (Greek: nine) in honor of Plotinus's Enneads
and Vervaeke's framework, the implementation has 10 dimensions due to the
four-fold structure of knowing (adding participatory to the original three).
The system maintains the triad-of-triads principle: three fundamental 
dimensions of reality (knowing, understanding, wisdom), each with its own
internal structure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict


# ============================================================================
# TRIAD I: WAYS OF KNOWING (Epistemological)
# ============================================================================

class KnowingMode(Enum):
    """The four fundamental ways of knowing"""
    PROPOSITIONAL = "propositional"    # Knowing-that (facts, theories)
    PROCEDURAL = "procedural"          # Knowing-how (skills, abilities)
    PERSPECTIVAL = "perspectival"      # Knowing-as (salience, framing)
    PARTICIPATORY = "participatory"     # Knowing-by-being (identity-constituting)


@dataclass
class KnowingDimension:
    """Represents one dimension of knowing with its activation and integration"""
    mode: KnowingMode
    activation: float = 0.5  # Current activation level [0,1]
    capacity: float = 1.0    # Maximum capacity
    integration_weights: Dict[KnowingMode, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.integration_weights:
            # Default integration weights - each mode influences others
            self.integration_weights = {
                KnowingMode.PROPOSITIONAL: 0.25,
                KnowingMode.PROCEDURAL: 0.25,
                KnowingMode.PERSPECTIVAL: 0.25,
                KnowingMode.PARTICIPATORY: 0.25
            }


# ============================================================================
# TRIAD II: ORDERS OF UNDERSTANDING (Ontological)
# ============================================================================

class UnderstandingOrder(Enum):
    """The three fundamental orders of understanding"""
    NOMOLOGICAL = "nomological"  # How things work (causal-scientific)
    NORMATIVE = "normative"      # What matters (evaluative-ethical)
    NARRATIVE = "narrative"      # How things develop (temporal-historical)


@dataclass
class OrderDimension:
    """Represents one order of understanding"""
    order: UnderstandingOrder
    coherence: float = 0.5      # How coherent this order is [0,1]
    salience: float = 0.5       # How salient this order is [0,1]
    patterns: List[Any] = field(default_factory=list)  # Detected patterns


# ============================================================================
# TRIAD III: PRACTICES OF WISDOM (Axiological)
# ============================================================================

class WisdomPractice(Enum):
    """The three fundamental practices of wisdom"""
    MORALITY = "morality"    # Virtue & character (ethical foundation)
    MEANING = "meaning"      # Coherence & purpose (existential fulfillment)
    MASTERY = "mastery"      # Excellence & flow (skilled engagement)


@dataclass
class WisdomDimension:
    """Represents one dimension of wisdom cultivation"""
    practice: WisdomPractice
    cultivation_level: float = 0.5  # Current cultivation [0,1]
    alignment: float = 0.5          # Alignment with other practices [0,1]
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# ENNEAD INTEGRATION CORE
# ============================================================================

@dataclass
class EnneadState:
    """Complete state of the Ennead system"""
    # Triad I: Ways of Knowing
    knowing_dimensions: Dict[KnowingMode, KnowingDimension] = field(default_factory=dict)
    
    # Triad II: Orders of Understanding
    order_dimensions: Dict[UnderstandingOrder, OrderDimension] = field(default_factory=dict)
    
    # Triad III: Practices of Wisdom
    wisdom_dimensions: Dict[WisdomPractice, WisdomDimension] = field(default_factory=dict)
    
    # Integration metrics
    triad_coherence: Dict[str, float] = field(default_factory=dict)
    overall_integration: float = 0.5
    
    def __post_init__(self):
        # Initialize knowing dimensions if not provided
        if not self.knowing_dimensions:
            for mode in KnowingMode:
                self.knowing_dimensions[mode] = KnowingDimension(mode=mode)
                
        # Initialize order dimensions if not provided
        if not self.order_dimensions:
            for order in UnderstandingOrder:
                self.order_dimensions[order] = OrderDimension(order=order)
                
        # Initialize wisdom dimensions if not provided
        if not self.wisdom_dimensions:
            for practice in WisdomPractice:
                self.wisdom_dimensions[practice] = WisdomDimension(practice=practice)


class EnneadRelevanceOptimizer:
    """
    Optimal relevance realization through Ennead integration.
    
    This system optimizes relevance realization by:
    1. Balancing all four ways of knowing
    2. Integrating nomological, normative, and narrative orders
    3. Cultivating morality, meaning, and mastery
    4. Achieving dynamic optimization across all nine dimensions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Ennead optimizer"""
        self.config = config or {}
        self.state = EnneadState()
        
        # Optimization parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.integration_threshold = self.config.get('integration_threshold', 0.7)
        self.sophrosyne_target = self.config.get('sophrosyne_target', 0.8)  # Optimal balance
        
        # History tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.relevance_episodes: List[Dict[str, Any]] = []
        
        # Meta-cognitive state
        self.meta_state = {
            'attention_focus': set(),
            'current_constraints': [],
            'optimization_trajectory': []
        }
        
    def realize_relevance(self, 
                         query: Set[Any], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main relevance realization function with Ennead optimization.
        
        Args:
            query: Set of items to evaluate for relevance
            context: Current contextual information
            
        Returns:
            Comprehensive relevance realization results
        """
        # Phase 1: Evaluate through Ways of Knowing
        knowing_results = self._realize_through_knowing(query, context)
        
        # Phase 2: Integrate through Orders of Understanding
        understanding_results = self._realize_through_understanding(
            knowing_results, context
        )
        
        # Phase 3: Optimize through Wisdom Practices
        wisdom_results = self._realize_through_wisdom(
            understanding_results, context
        )
        
        # Phase 4: Achieve Ennead Integration
        integrated_results = self._integrate_ennead(
            knowing_results, 
            understanding_results, 
            wisdom_results,
            context
        )
        
        # Phase 5: Optimize for Sophrosyne (optimal self-regulation)
        optimized_results = self._optimize_sophrosyne(integrated_results, context)
        
        # Track episode
        self._record_episode(query, context, optimized_results)
        
        return optimized_results
    
    def _realize_through_knowing(self, 
                                query: Set[Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Realize relevance through the four ways of knowing"""
        results = {}
        
        for mode, dimension in self.state.knowing_dimensions.items():
            mode_relevance = self._evaluate_knowing_mode(
                mode, dimension, query, context
            )
            results[mode.value] = mode_relevance
            
            # Update dimension activation based on relevance
            avg_relevance = np.mean(list(mode_relevance.values())) if mode_relevance else 0.0
            dimension.activation = self._lerp(
                dimension.activation, 
                avg_relevance, 
                self.learning_rate
            )
        
        # Compute knowing integration
        knowing_integration = self._compute_knowing_integration()
        results['knowing_integration'] = knowing_integration
        
        return results
    
    def _evaluate_knowing_mode(self,
                              mode: KnowingMode,
                              dimension: KnowingDimension,
                              query: Set[Any],
                              context: Dict[str, Any]) -> Dict[Any, float]:
        """Evaluate relevance through a specific knowing mode"""
        relevance_scores = {}
        
        for item in query:
            if mode == KnowingMode.PROPOSITIONAL:
                # Knowing-that: Evaluate based on facts and theories
                score = self._evaluate_propositional(item, context)
                
            elif mode == KnowingMode.PROCEDURAL:
                # Knowing-how: Evaluate based on skills and procedures
                score = self._evaluate_procedural(item, context)
                
            elif mode == KnowingMode.PERSPECTIVAL:
                # Knowing-as: Evaluate based on salience and framing
                score = self._evaluate_perspectival(item, context)
                
            elif mode == KnowingMode.PARTICIPATORY:
                # Knowing-by-being: Evaluate based on identity and transformation
                score = self._evaluate_participatory(item, context)
            else:
                score = 0.5
                
            # Modulate by dimension activation
            relevance_scores[item] = score * dimension.activation
            
        return relevance_scores
    
    def _evaluate_propositional(self, item: Any, context: Dict[str, Any]) -> float:
        """Evaluate propositional knowing (knowing-that)"""
        # Check if item is part of explicit knowledge
        item_str = str(item)
        
        # Factor 1: Explicit representation
        explicit_score = 1.0 if 'facts' in context and item_str in str(context['facts']) else 0.3
        
        # Factor 2: Logical coherence
        coherence_score = context.get('logical_coherence', 0.5)
        
        # Factor 3: Truth value
        truth_score = context.get('truth_values', {}).get(item_str, 0.5)
        
        return (explicit_score + coherence_score + truth_score) / 3.0
    
    def _evaluate_procedural(self, item: Any, context: Dict[str, Any]) -> float:
        """Evaluate procedural knowing (knowing-how)"""
        # Check if item relates to skills or procedures
        item_str = str(item)
        
        # Factor 1: Actionability
        action_score = 1.0 if 'actions' in context and item_str in str(context['actions']) else 0.3
        
        # Factor 2: Skill level
        skill_score = context.get('skill_levels', {}).get(item_str, 0.5)
        
        # Factor 3: Procedural fluency
        fluency_score = context.get('procedural_fluency', 0.5)
        
        return (action_score + skill_score + fluency_score) / 3.0
    
    def _evaluate_perspectival(self, item: Any, context: Dict[str, Any]) -> float:
        """Evaluate perspectival knowing (knowing-as)"""
        # Check salience and framing
        item_str = str(item)
        
        # Factor 1: Salience
        salience_score = context.get('salience_map', {}).get(item_str, 0.5)
        
        # Factor 2: Frame fit
        current_frame = context.get('current_frame', {})
        frame_score = 1.0 if item_str in str(current_frame) else 0.4
        
        # Factor 3: Aspect perception
        aspect_score = context.get('aspect_perception', 0.5)
        
        return (salience_score + frame_score + aspect_score) / 3.0
    
    def _evaluate_participatory(self, item: Any, context: Dict[str, Any]) -> float:
        """Evaluate participatory knowing (knowing-by-being)"""
        # Check identity and transformation relevance
        item_str = str(item)
        
        # Factor 1: Identity relevance
        identity_score = context.get('identity_relevance', {}).get(item_str, 0.5)
        
        # Factor 2: Transformative potential
        transform_score = context.get('transformative_potential', {}).get(item_str, 0.5)
        
        # Factor 3: Agapic connection
        connection_score = context.get('agapic_connection', 0.5)
        
        return (identity_score + transform_score + connection_score) / 3.0
    
    def _compute_knowing_integration(self) -> float:
        """Compute how well the four ways of knowing are integrated"""
        activations = [d.activation for d in self.state.knowing_dimensions.values()]
        
        # Integration is high when all modes are balanced (not too skewed)
        mean_activation = np.mean(activations)
        variance = np.var(activations)
        
        # Low variance indicates good balance
        balance_score = 1.0 / (1.0 + variance)
        
        # High mean indicates overall engagement
        engagement_score = mean_activation
        
        integration = 0.6 * balance_score + 0.4 * engagement_score
        return float(integration)
    
    def _realize_through_understanding(self,
                                      knowing_results: Dict[str, Any],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Realize relevance through the three orders of understanding"""
        results = {}
        
        for order, dimension in self.state.order_dimensions.items():
            order_analysis = self._analyze_order(
                order, dimension, knowing_results, context
            )
            results[order.value] = order_analysis
            
            # Update dimension coherence and salience
            dimension.coherence = order_analysis['coherence']
            dimension.salience = order_analysis['salience']
            dimension.patterns = order_analysis['patterns']
        
        # Compute understanding integration
        understanding_integration = self._compute_understanding_integration()
        results['understanding_integration'] = understanding_integration
        
        return results
    
    def _analyze_order(self,
                      order: UnderstandingOrder,
                      dimension: OrderDimension,
                      knowing_results: Dict[str, Any],
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific order of understanding"""
        
        if order == UnderstandingOrder.NOMOLOGICAL:
            # Analyze how things work (causal-scientific)
            return self._analyze_nomological(knowing_results, context)
            
        elif order == UnderstandingOrder.NORMATIVE:
            # Analyze what matters (evaluative-ethical)
            return self._analyze_normative(knowing_results, context)
            
        elif order == UnderstandingOrder.NARRATIVE:
            # Analyze how things develop (temporal-historical)
            return self._analyze_narrative(knowing_results, context)
        
        return {
            'coherence': 0.5,
            'salience': 0.5,
            'patterns': []
        }
    
    def _analyze_nomological(self, 
                           knowing_results: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nomological order (how things work)"""
        # Extract causal patterns
        patterns = []
        
        # Check for causal relationships in context
        if 'causal_links' in context:
            patterns.extend(context['causal_links'])
        
        # Procedural knowing provides mechanism insights
        procedural_relevance = knowing_results.get('procedural', {})
        if procedural_relevance:
            patterns.append({
                'type': 'procedural_mechanism',
                'items': list(procedural_relevance.keys()),
                'strength': np.mean(list(procedural_relevance.values()))
            })
        
        # Compute coherence: how well we understand mechanisms
        coherence = len(patterns) / max(1, context.get('expected_patterns', 5))
        coherence = min(1.0, coherence)
        
        # Compute salience: how relevant causal understanding is
        salience = context.get('nomological_salience', 0.6)
        
        return {
            'coherence': coherence,
            'salience': salience,
            'patterns': patterns
        }
    
    def _analyze_normative(self,
                          knowing_results: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze normative order (what matters)"""
        # Extract value patterns
        patterns = []
        
        # Check for values in context
        if 'values' in context:
            patterns.extend(context['values'])
        
        # Perspectival knowing provides salience
        perspectival_relevance = knowing_results.get('perspectival', {})
        if perspectival_relevance:
            patterns.append({
                'type': 'salience_structure',
                'items': list(perspectival_relevance.keys()),
                'strength': np.mean(list(perspectival_relevance.values()))
            })
        
        # Participatory knowing provides identity-based values
        participatory_relevance = knowing_results.get('participatory', {})
        if participatory_relevance:
            patterns.append({
                'type': 'identity_values',
                'items': list(participatory_relevance.keys()),
                'strength': np.mean(list(participatory_relevance.values()))
            })
        
        # Compute coherence: how well values are integrated
        coherence = len(patterns) / max(1, context.get('expected_values', 3))
        coherence = min(1.0, coherence)
        
        # Compute salience: how relevant normative understanding is
        salience = context.get('normative_salience', 0.7)
        
        return {
            'coherence': coherence,
            'salience': salience,
            'patterns': patterns
        }
    
    def _analyze_narrative(self,
                          knowing_results: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze narrative order (how things develop)"""
        # Extract temporal/developmental patterns
        patterns = []
        
        # Check for temporal sequences in context
        if 'temporal_sequences' in context:
            patterns.extend(context['temporal_sequences'])
        
        # Check history
        if 'history' in context:
            patterns.append({
                'type': 'historical_trajectory',
                'data': context['history'],
                'strength': 0.8
            })
        
        # Compute coherence: how well we understand development
        coherence = len(patterns) / max(1, context.get('expected_narrative', 2))
        coherence = min(1.0, coherence)
        
        # Compute salience: how relevant developmental understanding is
        salience = context.get('narrative_salience', 0.6)
        
        return {
            'coherence': coherence,
            'salience': salience,
            'patterns': patterns
        }
    
    def _compute_understanding_integration(self) -> float:
        """Compute how well the three orders are integrated"""
        # Integration requires all three orders
        coherences = [d.coherence for d in self.state.order_dimensions.values()]
        saliences = [d.salience for d in self.state.order_dimensions.values()]
        
        # Good integration means high coherence across all orders
        coherence_integration = np.mean(coherences)
        
        # And balanced salience (not overemphasizing one order)
        salience_balance = 1.0 / (1.0 + np.var(saliences))
        
        integration = 0.6 * coherence_integration + 0.4 * salience_balance
        return float(integration)
    
    def _realize_through_wisdom(self,
                               understanding_results: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Realize relevance through the three practices of wisdom"""
        results = {}
        
        for practice, dimension in self.state.wisdom_dimensions.items():
            practice_assessment = self._assess_wisdom_practice(
                practice, dimension, understanding_results, context
            )
            results[practice.value] = practice_assessment
            
            # Update dimension cultivation
            dimension.cultivation_level = practice_assessment['cultivation']
            dimension.alignment = practice_assessment['alignment']
            dimension.metrics = practice_assessment['metrics']
        
        # Compute wisdom integration
        wisdom_integration = self._compute_wisdom_integration()
        results['wisdom_integration'] = wisdom_integration
        
        return results
    
    def _assess_wisdom_practice(self,
                               practice: WisdomPractice,
                               dimension: WisdomDimension,
                               understanding_results: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess a specific wisdom practice"""
        
        if practice == WisdomPractice.MORALITY:
            # Assess virtue and character
            return self._assess_morality(understanding_results, context)
            
        elif practice == WisdomPractice.MEANING:
            # Assess coherence and purpose
            return self._assess_meaning(understanding_results, context)
            
        elif practice == WisdomPractice.MASTERY:
            # Assess excellence and flow
            return self._assess_mastery(understanding_results, context)
        
        return {
            'cultivation': 0.5,
            'alignment': 0.5,
            'metrics': {}
        }
    
    def _assess_morality(self,
                        understanding_results: Dict[str, Any],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess morality (virtue & character)"""
        # Morality derives primarily from normative order
        normative = understanding_results.get('normative', {})
        
        metrics = {
            'virtue_alignment': context.get('virtue_alignment', 0.5),
            'ethical_coherence': normative.get('coherence', 0.5),
            'compassion': context.get('compassion_level', 0.5)
        }
        
        cultivation = np.mean(list(metrics.values()))
        
        # Alignment with meaning and mastery
        alignment = 0.5  # Will be computed in integration
        
        return {
            'cultivation': cultivation,
            'alignment': alignment,
            'metrics': metrics
        }
    
    def _assess_meaning(self,
                       understanding_results: Dict[str, Any],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess meaning (coherence & purpose)"""
        # Meaning requires integration of all three orders
        nomological = understanding_results.get('nomological', {})
        normative = understanding_results.get('normative', {})
        narrative = understanding_results.get('narrative', {})
        
        metrics = {
            'coherence': (
                nomological.get('coherence', 0) + 
                normative.get('coherence', 0) + 
                narrative.get('coherence', 0)
            ) / 3.0,
            'purpose_clarity': context.get('purpose_clarity', 0.5),
            'significance': context.get('significance', 0.5),
            'connectedness': context.get('connectedness', 0.5)
        }
        
        cultivation = np.mean(list(metrics.values()))
        
        alignment = 0.5  # Will be computed in integration
        
        return {
            'cultivation': cultivation,
            'alignment': alignment,
            'metrics': metrics
        }
    
    def _assess_mastery(self,
                       understanding_results: Dict[str, Any],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess mastery (excellence & flow)"""
        # Mastery derives from procedural knowing and nomological understanding
        nomological = understanding_results.get('nomological', {})
        
        metrics = {
            'skill_level': context.get('skill_level', 0.5),
            'flow_state': context.get('flow_state', 0.5),
            'mechanism_understanding': nomological.get('coherence', 0.5),
            'performance_quality': context.get('performance_quality', 0.5)
        }
        
        cultivation = np.mean(list(metrics.values()))
        
        alignment = 0.5  # Will be computed in integration
        
        return {
            'cultivation': cultivation,
            'alignment': alignment,
            'metrics': metrics
        }
    
    def _compute_wisdom_integration(self) -> float:
        """Compute how well the three wisdom practices are integrated"""
        cultivations = [d.cultivation_level for d in self.state.wisdom_dimensions.values()]
        
        # Wisdom integration requires balance (sophrosyne)
        mean_cultivation = np.mean(cultivations)
        variance = np.var(cultivations)
        
        # Low variance indicates good balance
        balance_score = 1.0 / (1.0 + variance)
        
        # High mean indicates overall wisdom development
        development_score = mean_cultivation
        
        integration = 0.5 * balance_score + 0.5 * development_score
        return float(integration)
    
    def _integrate_ennead(self,
                         knowing_results: Dict[str, Any],
                         understanding_results: Dict[str, Any],
                         wisdom_results: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate all nine dimensions of the Ennead"""
        
        # Compute triad coherences
        triad_coherences = {
            'knowing': knowing_results.get('knowing_integration', 0.5),
            'understanding': understanding_results.get('understanding_integration', 0.5),
            'wisdom': wisdom_results.get('wisdom_integration', 0.5)
        }
        
        # Overall integration is the balanced combination
        overall_integration = np.mean(list(triad_coherences.values()))
        
        # Update state
        self.state.triad_coherence = triad_coherences
        self.state.overall_integration = overall_integration
        
        # Compute integrated relevance scores
        integrated_relevance = self._compute_integrated_relevance(
            knowing_results, understanding_results, wisdom_results
        )
        
        # Identify emergent patterns
        emergent_patterns = self._detect_emergent_patterns(
            knowing_results, understanding_results, wisdom_results
        )
        
        return {
            'integrated_relevance': integrated_relevance,
            'triad_coherences': triad_coherences,
            'overall_integration': overall_integration,
            'emergent_patterns': emergent_patterns,
            'knowing_results': knowing_results,
            'understanding_results': understanding_results,
            'wisdom_results': wisdom_results
        }
    
    def _compute_integrated_relevance(self,
                                     knowing_results: Dict[str, Any],
                                     understanding_results: Dict[str, Any],
                                     wisdom_results: Dict[str, Any]) -> Dict[Any, float]:
        """Compute final integrated relevance scores across all dimensions"""
        integrated_scores = defaultdict(float)
        item_counts = defaultdict(int)
        
        # Aggregate from knowing modes
        for mode in KnowingMode:
            mode_relevance = knowing_results.get(mode.value, {})
            for item, score in mode_relevance.items():
                integrated_scores[item] += score
                item_counts[item] += 1
        
        # Weight by understanding orders
        understanding_weight = understanding_results.get('understanding_integration', 0.5)
        
        # Weight by wisdom practices
        wisdom_weight = wisdom_results.get('wisdom_integration', 0.5)
        
        # Compute final scores
        final_relevance = {}
        for item, total_score in integrated_scores.items():
            count = item_counts[item]
            avg_score = total_score / count if count > 0 else 0.0
            
            # Apply understanding and wisdom weights
            final_score = avg_score * (0.4 + 0.3 * understanding_weight + 0.3 * wisdom_weight)
            final_relevance[item] = final_score
        
        return final_relevance
    
    def _detect_emergent_patterns(self,
                                 knowing_results: Dict[str, Any],
                                 understanding_results: Dict[str, Any],
                                 wisdom_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect emergent patterns across the Ennead"""
        patterns = []
        
        # Pattern 1: Knowing mode synergy
        knowing_integration = knowing_results.get('knowing_integration', 0)
        if knowing_integration > 0.8:
            patterns.append({
                'type': 'knowing_synergy',
                'strength': knowing_integration,
                'description': 'High integration across all ways of knowing'
            })
        
        # Pattern 2: Understanding coherence
        understanding_integration = understanding_results.get('understanding_integration', 0)
        if understanding_integration > 0.8:
            patterns.append({
                'type': 'understanding_coherence',
                'strength': understanding_integration,
                'description': 'Strong coherence across nomological, normative, narrative'
            })
        
        # Pattern 3: Wisdom balance
        wisdom_integration = wisdom_results.get('wisdom_integration', 0)
        if wisdom_integration > 0.8:
            patterns.append({
                'type': 'wisdom_balance',
                'strength': wisdom_integration,
                'description': 'Balanced cultivation of morality, meaning, mastery'
            })
        
        # Pattern 4: Ennead resonance (all three triads aligned)
        if all(x > 0.75 for x in [knowing_integration, understanding_integration, wisdom_integration]):
            patterns.append({
                'type': 'ennead_resonance',
                'strength': (knowing_integration + understanding_integration + wisdom_integration) / 3.0,
                'description': 'Full Ennead resonance - all nine dimensions integrated'
            })
        
        return patterns
    
    def _optimize_sophrosyne(self,
                            integrated_results: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize for sophrosyne (optimal self-regulation).
        
        Sophrosyne is the highest virtue - the optimal dynamic balance
        across all dimensions. Not static equilibrium but living optimization.
        """
        # Current integration level
        current_integration = integrated_results['overall_integration']
        
        # Compute gradient toward sophrosyne
        gradient = self.sophrosyne_target - current_integration
        
        # Apply optimization step
        if abs(gradient) > 0.01:  # Only optimize if significant deviation
            optimization = self._compute_sophrosyne_optimization(
                integrated_results, gradient, context
            )
            
            # Apply optimizations to state
            self._apply_optimizations(optimization)
            
            integrated_results['sophrosyne_optimization'] = optimization
        else:
            integrated_results['sophrosyne_optimization'] = {
                'status': 'optimal',
                'adjustments': []
            }
        
        # Compute final sophrosyne score
        sophrosyne_score = self._compute_sophrosyne_score(integrated_results)
        integrated_results['sophrosyne_score'] = sophrosyne_score
        
        return integrated_results
    
    def _compute_sophrosyne_optimization(self,
                                        integrated_results: Dict[str, Any],
                                        gradient: float,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optimizations needed to achieve sophrosyne"""
        optimizations = {
            'status': 'optimizing',
            'gradient': gradient,
            'adjustments': []
        }
        
        # Check each triad for imbalance
        triad_coherences = integrated_results['triad_coherences']
        
        for triad_name, coherence in triad_coherences.items():
            if coherence < self.sophrosyne_target - 0.1:
                # This triad needs strengthening
                optimizations['adjustments'].append({
                    'triad': triad_name,
                    'current': coherence,
                    'target': self.sophrosyne_target,
                    'action': 'strengthen',
                    'magnitude': (self.sophrosyne_target - coherence) * self.learning_rate
                })
            elif coherence > self.sophrosyne_target + 0.1:
                # This triad is overemphasized - balance needed
                optimizations['adjustments'].append({
                    'triad': triad_name,
                    'current': coherence,
                    'target': self.sophrosyne_target,
                    'action': 'balance',
                    'magnitude': (coherence - self.sophrosyne_target) * self.learning_rate
                })
        
        return optimizations
    
    def _apply_optimizations(self, optimization: Dict[str, Any]):
        """Apply sophrosyne optimizations to the state"""
        for adjustment in optimization['adjustments']:
            triad = adjustment['triad']
            magnitude = adjustment['magnitude']
            action = adjustment['action']
            
            if triad == 'knowing':
                # Adjust knowing dimensions
                for dimension in self.state.knowing_dimensions.values():
                    if action == 'strengthen':
                        dimension.activation = min(1.0, dimension.activation + magnitude)
                    elif action == 'balance':
                        dimension.activation = max(0.0, dimension.activation - magnitude * 0.5)
            
            elif triad == 'understanding':
                # Adjust understanding dimensions
                for dimension in self.state.order_dimensions.values():
                    if action == 'strengthen':
                        dimension.salience = min(1.0, dimension.salience + magnitude)
                    elif action == 'balance':
                        dimension.salience = max(0.0, dimension.salience - magnitude * 0.5)
            
            elif triad == 'wisdom':
                # Adjust wisdom dimensions
                for dimension in self.state.wisdom_dimensions.values():
                    if action == 'strengthen':
                        dimension.cultivation_level = min(1.0, dimension.cultivation_level + magnitude)
                    elif action == 'balance':
                        dimension.cultivation_level = max(0.0, dimension.cultivation_level - magnitude * 0.5)
    
    def _compute_sophrosyne_score(self, integrated_results: Dict[str, Any]) -> float:
        """
        Compute sophrosyne score (optimal self-regulation).
        
        High sophrosyne means:
        1. Balance across all triads
        2. High overall integration
        3. Dynamic responsiveness (not rigid)
        """
        triad_coherences = integrated_results['triad_coherences']
        overall_integration = integrated_results['overall_integration']
        
        # Factor 1: Balance (low variance across triads)
        coherence_values = list(triad_coherences.values())
        balance_score = 1.0 / (1.0 + np.var(coherence_values))
        
        # Factor 2: Integration (high mean)
        integration_score = overall_integration
        
        # Factor 3: Proximity to target
        proximity_score = 1.0 - abs(overall_integration - self.sophrosyne_target)
        
        # Weighted combination
        sophrosyne = (
            0.4 * balance_score +
            0.3 * integration_score +
            0.3 * proximity_score
        )
        
        return sophrosyne
    
    def _record_episode(self,
                       query: Set[Any],
                       context: Dict[str, Any],
                       results: Dict[str, Any]):
        """Record a relevance realization episode for learning"""
        episode = {
            'query': query,
            'context': context,
            'results': results,
            'state': {
                'knowing': {k.value: v.activation for k, v in self.state.knowing_dimensions.items()},
                'understanding': {k.value: v.coherence for k, v in self.state.order_dimensions.items()},
                'wisdom': {k.value: v.cultivation_level for k, v in self.state.wisdom_dimensions.items()},
                'overall_integration': self.state.overall_integration
            }
        }
        
        self.relevance_episodes.append(episode)
        
        # Trim history if too long
        if len(self.relevance_episodes) > 1000:
            self.relevance_episodes = self.relevance_episodes[-1000:]
    
    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation"""
        return a + (b - a) * t
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current Ennead state"""
        return {
            'knowing': {
                mode.value: {
                    'activation': dim.activation,
                    'capacity': dim.capacity
                }
                for mode, dim in self.state.knowing_dimensions.items()
            },
            'understanding': {
                order.value: {
                    'coherence': dim.coherence,
                    'salience': dim.salience,
                    'pattern_count': len(dim.patterns)
                }
                for order, dim in self.state.order_dimensions.items()
            },
            'wisdom': {
                practice.value: {
                    'cultivation': dim.cultivation_level,
                    'alignment': dim.alignment,
                    'metrics': dim.metrics
                }
                for practice, dim in self.state.wisdom_dimensions.items()
            },
            'integration': {
                'triad_coherences': self.state.triad_coherence,
                'overall': self.state.overall_integration
            },
            'episodes_recorded': len(self.relevance_episodes)
        }
