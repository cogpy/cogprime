"""
Integration Adapter for Ennead Relevance Optimizer

This module provides adapters to integrate the Ennead Relevance Optimizer
with existing relevance realization systems in CogPrime.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from .ennead_relevance_optimizer import EnneadRelevanceOptimizer
except ImportError:
    from ennead_relevance_optimizer import EnneadRelevanceOptimizer


@dataclass
class RelevanceResults:
    """Unified relevance results from multiple systems"""
    ennead_results: Dict[str, Any]
    legacy_results: Optional[Dict[str, Any]] = None
    integrated_scores: Optional[Dict[Any, float]] = None
    meta_information: Optional[Dict[str, Any]] = None


class EnneadRelevanceAdapter:
    """
    Adapter for integrating Ennead Relevance Optimizer with existing systems.
    
    This adapter provides:
    1. Context enrichment from existing relevance data
    2. Results translation between systems
    3. Unified relevance scoring
    4. Backward compatibility
    """
    
    def __init__(self, 
                 ennead_config: Optional[Dict] = None,
                 enable_legacy: bool = True):
        """
        Initialize the adapter.
        
        Args:
            ennead_config: Configuration for Ennead optimizer
            enable_legacy: Whether to support legacy systems
        """
        self.ennead_optimizer = EnneadRelevanceOptimizer(config=ennead_config)
        self.enable_legacy = enable_legacy
        
        # Context enrichment strategies
        self.enrichment_strategies = {
            'knowing': self._enrich_knowing_context,
            'understanding': self._enrich_understanding_context,
            'wisdom': self._enrich_wisdom_context
        }
        
    def realize_relevance(self,
                         query: Set[Any],
                         context: Dict[str, Any],
                         use_legacy: bool = False) -> RelevanceResults:
        """
        Unified relevance realization using Ennead optimizer.
        
        Args:
            query: Set of items to evaluate
            context: Current context
            use_legacy: Whether to also compute legacy results
            
        Returns:
            Unified relevance results
        """
        # Enrich context for Ennead optimizer
        enriched_context = self._enrich_context(context)
        
        # Compute Ennead results
        ennead_results = self.ennead_optimizer.realize_relevance(
            query, enriched_context
        )
        
        # Compute legacy results if requested
        legacy_results = None
        if use_legacy and self.enable_legacy:
            legacy_results = self._compute_legacy_results(query, context)
        
        # Integrate scores
        integrated_scores = self._integrate_scores(
            ennead_results, legacy_results
        )
        
        # Gather meta information
        meta_info = self._gather_meta_information(
            ennead_results, legacy_results
        )
        
        return RelevanceResults(
            ennead_results=ennead_results,
            legacy_results=legacy_results,
            integrated_scores=integrated_scores,
            meta_information=meta_info
        )
    
    def _enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich context with data needed for all Ennead dimensions.
        
        This method fills in missing context elements by:
        1. Inferring from available data
        2. Using defaults when inference not possible
        3. Extracting implicit information
        """
        enriched = context.copy()
        
        # Enrich knowing dimensions
        enriched = self._enrich_knowing_context(enriched)
        
        # Enrich understanding dimensions
        enriched = self._enrich_understanding_context(enriched)
        
        # Enrich wisdom dimensions
        enriched = self._enrich_wisdom_context(enriched)
        
        return enriched
    
    def _enrich_knowing_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich context for ways of knowing"""
        
        # Propositional knowing enrichment
        if 'facts' not in context:
            # Infer facts from other data
            facts = []
            if 'knowledge_base' in context:
                facts = list(context['knowledge_base'].keys())
            context['facts'] = facts
        
        if 'logical_coherence' not in context:
            # Default coherence based on fact count
            fact_count = len(context.get('facts', []))
            context['logical_coherence'] = min(1.0, 0.5 + fact_count * 0.05)
        
        # Procedural knowing enrichment
        if 'actions' not in context:
            # Infer actions from capabilities
            actions = []
            if 'capabilities' in context:
                actions = list(context['capabilities'])
            context['actions'] = actions
        
        if 'skill_levels' not in context:
            # Default skill levels
            context['skill_levels'] = {
                action: 0.5 for action in context.get('actions', [])
            }
        
        if 'procedural_fluency' not in context:
            # Average skill level
            skill_levels = context.get('skill_levels', {})
            if skill_levels:
                context['procedural_fluency'] = np.mean(list(skill_levels.values()))
            else:
                context['procedural_fluency'] = 0.5
        
        # Perspectival knowing enrichment
        if 'salience_map' not in context:
            # Create salience map from attention data
            salience_map = {}
            if 'attention' in context:
                salience_map = context['attention']
            elif 'importance' in context:
                salience_map = context['importance']
            context['salience_map'] = salience_map
        
        if 'current_frame' not in context:
            # Infer frame from goals or focus
            if 'goals' in context:
                context['current_frame'] = f"goal_oriented: {context['goals']}"
            else:
                context['current_frame'] = "default_frame"
        
        if 'aspect_perception' not in context:
            # Default aspect perception
            context['aspect_perception'] = 0.6
        
        # Participatory knowing enrichment
        if 'identity_relevance' not in context:
            # Infer from values or commitments
            identity_relevance = {}
            if 'values' in context:
                for value in context['values']:
                    if isinstance(value, dict) and 'type' in value:
                        identity_relevance[value['type']] = value.get('strength', 0.5)
            context['identity_relevance'] = identity_relevance
        
        if 'transformative_potential' not in context:
            # Default transformative potential
            context['transformative_potential'] = {}
        
        if 'agapic_connection' not in context:
            # Default agapic connection
            context['agapic_connection'] = 0.5
        
        return context
    
    def _enrich_understanding_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich context for orders of understanding"""
        
        # Nomological order enrichment
        if 'causal_links' not in context:
            # Infer causal links from relationships
            causal_links = []
            if 'relationships' in context:
                for rel in context['relationships']:
                    if isinstance(rel, dict):
                        if 'cause' in rel and 'effect' in rel:
                            causal_links.append({
                                'from': rel['cause'],
                                'to': rel['effect']
                            })
            context['causal_links'] = causal_links
        
        if 'nomological_salience' not in context:
            # Default based on causal link count
            link_count = len(context.get('causal_links', []))
            context['nomological_salience'] = min(1.0, 0.5 + link_count * 0.1)
        
        # Normative order enrichment
        if 'values' not in context:
            # Infer values from goals or preferences
            values = []
            if 'goals' in context:
                for goal in context['goals']:
                    values.append({'type': f'goal_{goal}', 'strength': 0.7})
            context['values'] = values
        
        if 'normative_salience' not in context:
            # Default based on value count
            value_count = len(context.get('values', []))
            context['normative_salience'] = min(1.0, 0.6 + value_count * 0.05)
        
        # Narrative order enrichment
        if 'temporal_sequences' not in context:
            # Infer from history or events
            sequences = []
            if 'history' in context:
                sequences.append({'events': context['history']})
            context['temporal_sequences'] = sequences
        
        if 'narrative_salience' not in context:
            # Default based on temporal data
            seq_count = len(context.get('temporal_sequences', []))
            context['narrative_salience'] = min(1.0, 0.5 + seq_count * 0.15)
        
        return context
    
    def _enrich_wisdom_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich context for practices of wisdom"""
        
        # Morality enrichment
        if 'virtue_alignment' not in context:
            # Infer from ethical considerations
            virtue_alignment = 0.5
            if 'ethics' in context:
                virtue_alignment = context['ethics'].get('alignment', 0.5)
            context['virtue_alignment'] = virtue_alignment
        
        if 'compassion_level' not in context:
            # Default compassion
            context['compassion_level'] = 0.6
        
        # Meaning enrichment
        if 'purpose_clarity' not in context:
            # Infer from goals
            purpose_clarity = 0.5
            if 'goals' in context and context['goals']:
                purpose_clarity = 0.7
            context['purpose_clarity'] = purpose_clarity
        
        if 'significance' not in context:
            # Default significance
            context['significance'] = 0.6
        
        if 'connectedness' not in context:
            # Infer from social connections
            connectedness = 0.5
            if 'social_connections' in context:
                connectedness = min(1.0, len(context['social_connections']) * 0.1)
            context['connectedness'] = connectedness
        
        # Mastery enrichment
        if 'skill_level' not in context:
            # Use procedural fluency as proxy
            context['skill_level'] = context.get('procedural_fluency', 0.5)
        
        if 'flow_state' not in context:
            # Infer from engagement
            flow_state = 0.5
            if 'engagement' in context:
                flow_state = context['engagement']
            context['flow_state'] = flow_state
        
        if 'performance_quality' not in context:
            # Infer from skill level
            context['performance_quality'] = context.get('skill_level', 0.5)
        
        return context
    
    def _compute_legacy_results(self, 
                               query: Set[Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute results using legacy relevance systems.
        
        This provides backward compatibility with existing code.
        """
        # This would integrate with existing RelevanceCore, etc.
        # For now, return a placeholder structure
        return {
            'relevant_items': set(),
            'confidence': 0.0,
            'method': 'legacy'
        }
    
    def _integrate_scores(self,
                         ennead_results: Dict[str, Any],
                         legacy_results: Optional[Dict[str, Any]]) -> Dict[Any, float]:
        """
        Integrate relevance scores from multiple systems.
        
        This creates a unified relevance score that combines:
        1. Ennead integrated relevance
        2. Legacy system scores (if available)
        3. Meta-cognitive weighting
        """
        # Start with Ennead scores
        integrated = ennead_results.get('integrated_relevance', {}).copy()
        
        # Blend with legacy scores if available
        if legacy_results and 'relevant_items' in legacy_results:
            # Give more weight to Ennead (0.8) vs legacy (0.2)
            for item in legacy_results['relevant_items']:
                legacy_score = 0.5  # Default score
                if item in integrated:
                    integrated[item] = 0.8 * integrated[item] + 0.2 * legacy_score
                else:
                    integrated[item] = 0.2 * legacy_score
        
        # Apply sophrosyne weighting
        sophrosyne_score = ennead_results.get('sophrosyne_score', 0.8)
        for item in integrated:
            integrated[item] *= (0.5 + 0.5 * sophrosyne_score)
        
        return integrated
    
    def _gather_meta_information(self,
                                ennead_results: Dict[str, Any],
                                legacy_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Gather meta information about the relevance realization"""
        meta = {
            'system': 'ennead_integrated',
            'ennead_integration': ennead_results.get('overall_integration', 0),
            'sophrosyne_score': ennead_results.get('sophrosyne_score', 0),
            'emergent_patterns': len(ennead_results.get('emergent_patterns', [])),
            'triad_coherences': ennead_results.get('triad_coherences', {}),
            'used_legacy': legacy_results is not None
        }
        
        # Add pattern details
        patterns = ennead_results.get('emergent_patterns', [])
        if patterns:
            meta['patterns'] = [
                {'type': p['type'], 'strength': p['strength']}
                for p in patterns
            ]
        
        return meta
    
    def get_ennead_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive Ennead state summary"""
        return self.ennead_optimizer.get_state_summary()
    
    def adapt_for_opencog(self, 
                          ennead_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt Ennead results for OpenCog AtomSpace integration.
        
        Returns attention allocation parameters for OpenCog.
        """
        sophrosyne = ennead_results.get('sophrosyne_score', 0.8)
        integration = ennead_results.get('overall_integration', 0.7)
        
        return {
            'sti_boost_factor': 10.0 * sophrosyne,
            'lti_boost_factor': 5.0 * integration,
            'attention_threshold': 0.1 / sophrosyne if sophrosyne > 0 else 0.1,
            'focus_items': list(ennead_results.get('integrated_relevance', {}).keys())[:10]
        }
    
    def adapt_for_action_generation(self,
                                   ennead_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt Ennead results for action generation.
        
        Returns action-relevant data based on Ennead wisdom practices.
        """
        wisdom_results = ennead_results.get('wisdom_results', {})
        
        morality = wisdom_results.get('morality', {})
        meaning = wisdom_results.get('meaning', {})
        mastery = wisdom_results.get('mastery', {})
        
        return {
            'ethical_constraints': {
                'virtue_threshold': morality.get('cultivation', 0.5),
                'compassion_required': morality.get('metrics', {}).get('compassion', 0.5)
            },
            'purpose_alignment': {
                'coherence_required': meaning.get('cultivation', 0.5),
                'significance_threshold': meaning.get('metrics', {}).get('significance', 0.5)
            },
            'skill_requirements': {
                'mastery_level': mastery.get('cultivation', 0.5),
                'flow_target': mastery.get('metrics', {}).get('flow_state', 0.5)
            }
        }
    
    def adapt_for_learning(self,
                          ennead_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt Ennead results for learning systems.
        
        Returns learning targets based on Ennead integration gaps.
        """
        triad_coherences = ennead_results.get('triad_coherences', {})
        
        # Identify weakest triad
        weakest_triad = min(triad_coherences.items(), key=lambda x: x[1])
        
        # Identify optimization priorities
        sophrosyne_opt = ennead_results.get('sophrosyne_optimization', {})
        adjustments = sophrosyne_opt.get('adjustments', [])
        
        return {
            'weakest_triad': weakest_triad[0],
            'weakest_coherence': weakest_triad[1],
            'optimization_priorities': [
                {
                    'dimension': adj.get('triad'),
                    'action': adj.get('action'),
                    'magnitude': adj.get('magnitude')
                }
                for adj in adjustments
            ],
            'learning_targets': {
                'knowing_integration': triad_coherences.get('knowing', 0.5),
                'understanding_integration': triad_coherences.get('understanding', 0.5),
                'wisdom_integration': triad_coherences.get('wisdom', 0.5)
            }
        }


class SimpleEnneadInterface:
    """
    Simplified interface for basic Ennead relevance realization.
    
    This provides a straightforward API for common use cases.
    """
    
    def __init__(self):
        self.adapter = EnneadRelevanceAdapter()
    
    def find_relevant(self,
                     items: List[Any],
                     context: Optional[Dict[str, Any]] = None) -> List[Tuple[Any, float]]:
        """
        Find relevant items with scores.
        
        Args:
            items: Items to evaluate
            context: Optional context (will be enriched automatically)
            
        Returns:
            List of (item, relevance_score) tuples, sorted by relevance
        """
        query = set(items)
        context = context or {}
        
        results = self.adapter.realize_relevance(query, context)
        
        # Extract scores
        scores = results.integrated_scores or {}
        
        # Sort by relevance
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_items
    
    def get_integration_level(self) -> float:
        """Get current Ennead integration level"""
        summary = self.adapter.get_ennead_state_summary()
        return summary['integration']['overall']
    
    def get_wisdom_balance(self) -> Dict[str, float]:
        """Get current wisdom practice balance"""
        summary = self.adapter.get_ennead_state_summary()
        return {
            practice: data['cultivation']
            for practice, data in summary['wisdom'].items()
        }


# Example usage functions

def example_basic_usage():
    """Example: Basic relevance realization"""
    interface = SimpleEnneadInterface()
    
    items = ['concept_a', 'skill_b', 'value_c']
    context = {
        'goals': ['learn', 'grow'],
        'capabilities': ['analyze', 'synthesize'],
        'values': [{'type': 'truth', 'strength': 0.9}]
    }
    
    relevant_items = interface.find_relevant(items, context)
    
    print("Relevant items:")
    for item, score in relevant_items:
        print(f"  {item}: {score:.3f}")
    
    print(f"\nIntegration level: {interface.get_integration_level():.3f}")
    print(f"Wisdom balance: {interface.get_wisdom_balance()}")


def example_advanced_usage():
    """Example: Advanced usage with full adapter"""
    adapter = EnneadRelevanceAdapter(ennead_config={
        'learning_rate': 0.05,
        'sophrosyne_target': 0.85
    })
    
    query = {'item_x', 'item_y', 'item_z'}
    context = {
        'facts': ['item_x is important'],
        'skill_level': 0.7,
        'virtue_alignment': 0.8
    }
    
    results = adapter.realize_relevance(query, context, use_legacy=False)
    
    print("Ennead Results:")
    print(f"  Integration: {results.meta_information['ennead_integration']:.3f}")
    print(f"  Sophrosyne: {results.meta_information['sophrosyne_score']:.3f}")
    
    # Adapt for different systems
    opencog_params = adapter.adapt_for_opencog(results.ennead_results)
    print(f"\nOpenCog STI boost: {opencog_params['sti_boost_factor']:.2f}")
    
    action_params = adapter.adapt_for_action_generation(results.ennead_results)
    print(f"Virtue threshold: {action_params['ethical_constraints']['virtue_threshold']:.2f}")


if __name__ == '__main__':
    print("Example 1: Basic Usage")
    print("=" * 60)
    example_basic_usage()
    
    print("\n\nExample 2: Advanced Usage")
    print("=" * 60)
    example_advanced_usage()
