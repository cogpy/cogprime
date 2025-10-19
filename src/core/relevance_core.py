from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import numpy as np
import math
from dataclasses import dataclass
from ..atomspace import AtomSpace, Node, Link, TruthValue, AttentionValue

class RelevanceMode(Enum):
    """Different modes of relevance realization"""
    SELECTIVE_ATTENTION = "selective_attention"  # Bottom-up salience
    WORKING_MEMORY = "working_memory"  # Active maintenance
    PROBLEM_SPACE = "problem_space"  # Search space navigation
    SIDE_EFFECTS = "side_effects"  # Action consequences
    LONG_TERM_MEMORY = "long_term_memory"  # Organization & access

class RelevanceCore:
    """Core relevance realization system that implements Vervaeke's framework.
    
    This system coordinates multiple interacting relevance modes to enable
    intelligent behavior through dynamic self-organization of salience landscapes.
    """
    
    def __init__(self):
        # Salience weights for different modes
        self.mode_weights: Dict[RelevanceMode, float] = {
            mode: 1.0 for mode in RelevanceMode
        }
        
        # Current active contents
        self.active_contents: Dict[RelevanceMode, Set] = {
            mode: set() for mode in RelevanceMode
        }
        
        # Salience thresholds for filtering
        self.thresholds: Dict[RelevanceMode, float] = {
            mode: 0.5 for mode in RelevanceMode
        }
        
        # Interaction weights between modes
        self.interaction_weights = np.ones((len(RelevanceMode), len(RelevanceMode)))
        
    def update_salience(self, mode: RelevanceMode, contents: Set, 
                       context: Optional[Dict] = None) -> Set:
        """Update salience weights for given contents in a mode.
        
        Args:
            mode: The relevance mode to update
            contents: Set of items to evaluate
            context: Optional contextual information
            
        Returns:
            Set of items above salience threshold
        """
        # Get base salience for contents
        salience = self._compute_base_salience(contents, context)
        
        # Modulate by mode interactions
        for other_mode in RelevanceMode:
            if other_mode != mode:
                interaction_weight = self.interaction_weights[
                    mode.value, other_mode.value
                ]
                other_contents = self.active_contents[other_mode]
                salience = self._modulate_salience(
                    salience, other_contents, interaction_weight
                )
                
        # Filter by threshold
        threshold = self.thresholds[mode]
        salient_items = {
            item for item, weight in salience.items() 
            if weight >= threshold
        }
        
        # Update active contents
        self.active_contents[mode] = salient_items
        
        return salient_items
    
    def _compute_base_salience(self, contents: Set, 
                             context: Optional[Dict]) -> Dict:
        """Compute base salience weights for contents."""
        # Placeholder for more sophisticated salience computation
        return {item: np.random.random() for item in contents}
        
    def _modulate_salience(self, salience: Dict, other_contents: Set,
                          interaction_weight: float) -> Dict:
        """Modulate salience based on contents in other modes."""
        # Placeholder for more sophisticated interaction
        return {
            k: v * interaction_weight 
            for k, v in salience.items()
        }
        
    def restructure_salience(self, mode: RelevanceMode,
                           new_context: Dict) -> None:
        """Dynamically restructure salience landscape based on new context.
        
        This implements the insight/reframing mechanism described by Vervaeke.
        """
        # Update thresholds based on context
        self.thresholds[mode] *= new_context.get('threshold_mod', 1.0)
        
        # Update interaction weights
        for other_mode in RelevanceMode:
            if other_mode != mode:
                self.interaction_weights[mode.value, other_mode.value] *= \
                    new_context.get('interaction_mod', 1.0)
                    
        # Re-evaluate active contents with new parameters
        self.update_salience(mode, self.active_contents[mode], new_context)

    def evaluate_relevance(self, query: Set, context: Dict) -> Tuple[Set, float]:
        """Evaluate relevance of query items in current context.
        
        Returns both relevant items and confidence score.
        """
        relevant_items = set()
        total_confidence = 0.0
        
        # Check relevance across all modes
        for mode in RelevanceMode:
            # Update salience landscape
            salient = self.update_salience(mode, query, context)
            
            # Accumulate results
            relevant_items.update(salient)
            total_confidence += len(salient) / len(query)
            
        # Normalize confidence
        confidence = total_confidence / len(RelevanceMode)
        
        return relevant_items, confidence


@dataclass
class SalienceRecord:
    """Record of salience information for tracking dynamics."""
    timestamp: float
    salience_value: float
    context: Dict[str, Any]
    mode: RelevanceMode


class OpenCogRelevanceEngine:
    """
    OpenCog-style relevance realization engine implementing Vervaeke's framework
    with AtomSpace integration and Silicon Sage capabilities.
    
    This engine combines:
    - OpenCog AtomSpace for knowledge representation
    - Vervaeke's relevance realization framework
    - Dynamic attention allocation
    - Meta-cognitive monitoring
    """
    
    def __init__(self, atomspace: AtomSpace = None, config: Dict = None):
        """Initialize the OpenCog relevance engine.
        
        Args:
            atomspace: Optional AtomSpace instance
            config: Configuration parameters
        """
        self.config = config or {}
        self.atomspace = atomspace or AtomSpace()
        
        # Core relevance system
        self.relevance_core = RelevanceCore()
        
        # Attention allocation system
        self.attention_bank = AttentionBank(
            atomspace=self.atomspace,
            total_sti=self.config.get('total_sti', 1000.0),
            total_lti=self.config.get('total_lti', 1000.0)
        )
        
        # Salience tracking
        self.salience_history: List[SalienceRecord] = []
        self.current_focus = set()
        
        # Meta-cognitive components
        self.meta_patterns = {}
        self.emergent_behaviors = set()
        
        # Initialize core concepts in AtomSpace
        self._initialize_core_concepts()
        
        # Pattern matcher for inference
        self.pattern_matcher = OpenCogPatternMatcher(self.atomspace)
        
        # Meaning-making systems
        self.meaning_maker = MeaningMaker(self.atomspace)
        
    def _initialize_core_concepts(self):
        """Initialize core relevance concepts in the AtomSpace."""
        # Relevance concepts
        relevance_node = Node("ConceptNode", "Relevance")
        self.atomspace.add(relevance_node)
        
        # Mode concepts
        for mode in RelevanceMode:
            mode_node = Node("ConceptNode", f"RelevanceMode_{mode.value}")
            self.atomspace.add(mode_node)
            
            # Link mode to relevance
            mode_link = Link("InheritanceLink", [mode_node, relevance_node])
            self.atomspace.add(mode_link)
        
        # Salience concept
        salience_node = Node("ConceptNode", "Salience")
        self.atomspace.add(salience_node)
        
        # Attention concepts
        attention_node = Node("ConceptNode", "Attention")
        self.atomspace.add(attention_node)
        
        # Create foundational relationships
        relevance_attention_link = Link("AssociativeLink", [relevance_node, attention_node])
        relevance_attention_link.set_truth_value(TruthValue(0.9, 0.8))
        self.atomspace.add(relevance_attention_link)
        
    def realize_relevance(self, query_atoms: List[Any], context: Dict = None) -> Dict[str, Any]:
        """
        Main relevance realization function implementing Silicon Sage capabilities.
        
        Args:
            query_atoms: List of atoms or concepts to evaluate relevance for
            context: Current contextual information
            
        Returns:
            Dictionary with relevance results and meta-information
        """
        context = context or {}
        
        # Convert query to atom representations
        query_set = self._prepare_query_atoms(query_atoms)
        
        # Update attention values based on current relevance
        self._update_attention_values(query_set, context)
        
        # Perform relevance evaluation across all modes
        relevance_results = {}
        
        for mode in RelevanceMode:
            mode_result = self._evaluate_mode_relevance(mode, query_set, context)
            relevance_results[mode.value] = mode_result
            
        # Integrate results using meta-cognitive processes
        integrated_result = self._integrate_relevance_results(relevance_results, context)
        
        # Track salience dynamics
        self._track_salience_dynamics(integrated_result, context)
        
        # Detect emergent patterns
        emergent_patterns = self._detect_emergent_patterns(integrated_result)
        
        # Update focus of attention
        self._update_attention_focus(integrated_result)
        
        return {
            'relevant_atoms': integrated_result['relevant_atoms'],
            'salience_landscape': integrated_result['salience_landscape'],
            'attention_allocation': integrated_result['attention_allocation'],
            'emergent_patterns': emergent_patterns,
            'meaning_structures': self.meaning_maker.extract_meaning(
                integrated_result['relevant_atoms'], context
            ),
            'meta_cognitive_state': self._get_meta_cognitive_state(),
            'confidence': integrated_result['confidence']
        }
        
    def _prepare_query_atoms(self, query_atoms: List[Any]) -> Set[Any]:
        """Prepare query atoms for relevance evaluation."""
        query_set = set()
        
        for item in query_atoms:
            if isinstance(item, str):
                # Create concept node for string
                concept_node = Node("ConceptNode", item)
                self.atomspace.add(concept_node)
                query_set.add(concept_node)
            elif hasattr(item, 'atom_type'):
                # Already an atom
                query_set.add(item)
            else:
                # Convert to string representation
                concept_node = Node("ConceptNode", str(item))
                self.atomspace.add(concept_node)
                query_set.add(concept_node)
                
        return query_set
        
    def _update_attention_values(self, query_set: Set[Any], context: Dict):
        """Update attention values in the AtomSpace based on relevance."""
        for atom in query_set:
            if hasattr(atom, 'get_attention_value'):
                current_av = atom.get_attention_value()
                
                # Boost STI for currently relevant atoms
                new_sti = current_av.sti + context.get('sti_boost', 10.0)
                new_av = AttentionValue(new_sti, current_av.lti, current_av.vlti)
                atom.set_attention_value(new_av)
                
                # Update attention bank
                self.attention_bank.update_atom_attention(atom, new_av)
                
    def _evaluate_mode_relevance(self, mode: RelevanceMode, query_set: Set[Any], 
                               context: Dict) -> Dict[str, Any]:
        """Evaluate relevance for a specific mode."""
        # Use relevance core for basic evaluation
        relevant_items, confidence = self.relevance_core.evaluate_relevance(query_set, context)
        
        # Enhance with OpenCog pattern matching
        patterns = self._generate_relevance_patterns(mode, query_set)
        pattern_matches = []
        
        for pattern in patterns:
            matches = self.pattern_matcher.match_pattern(pattern)
            pattern_matches.extend(matches)
            
        # Combine results
        enhanced_relevant = relevant_items.union(set(pattern_matches))
        
        # Compute salience landscape for this mode
        salience_landscape = self._compute_mode_salience_landscape(
            mode, enhanced_relevant, context
        )
        
        return {
            'relevant_items': enhanced_relevant,
            'salience_landscape': salience_landscape,
            'confidence': confidence,
            'pattern_matches': pattern_matches
        }
        
    def _generate_relevance_patterns(self, mode: RelevanceMode, query_set: Set[Any]) -> List[Any]:
        """Generate patterns for relevance matching based on mode."""
        patterns = []
        
        if mode == RelevanceMode.SELECTIVE_ATTENTION:
            # Look for associations and similarities
            for atom in query_set:
                if hasattr(atom, 'name'):
                    # Pattern: (AssociativeLink $X atom)
                    pattern = Link("AssociativeLink", [None, atom])
                    patterns.append(pattern)
                    
        elif mode == RelevanceMode.WORKING_MEMORY:
            # Look for active links and recent patterns
            for atom in query_set:
                # Pattern: (InheritanceLink atom $Y)
                pattern = Link("InheritanceLink", [atom, None])
                patterns.append(pattern)
                
        elif mode == RelevanceMode.PROBLEM_SPACE:
            # Look for goal-relevant patterns
            goal_node = Node("ConceptNode", "Goal")
            for atom in query_set:
                # Pattern: (EvaluationLink Goal (ListLink atom $X))
                pattern = Link("EvaluationLink", [
                    goal_node,
                    Link("ListLink", [atom, None])
                ])
                patterns.append(pattern)
                
        # Add more mode-specific patterns...
        
        return patterns
        
    def _compute_mode_salience_landscape(self, mode: RelevanceMode, 
                                       relevant_items: Set[Any], 
                                       context: Dict) -> Dict[Any, float]:
        """Compute salience landscape for a specific mode."""
        landscape = {}
        
        for item in relevant_items:
            # Base salience from attention value
            base_salience = 0.5
            if hasattr(item, 'get_attention_value'):
                av = item.get_attention_value()
                base_salience = self._sti_to_salience(av.sti)
                
            # Modulate by mode-specific factors
            mode_modifier = self._get_mode_modifier(mode, item, context)
            
            # Compute final salience
            salience = base_salience * mode_modifier
            landscape[item] = salience
            
        return landscape
        
    def _sti_to_salience(self, sti: float) -> float:
        """Convert STI (Short-Term Importance) to salience value."""
        return 1.0 / (1.0 + math.exp(-sti / 100.0))  # Sigmoid transformation
        
    def _get_mode_modifier(self, mode: RelevanceMode, item: Any, context: Dict) -> float:
        """Get mode-specific modifier for salience computation."""
        modifier = 1.0
        
        if mode == RelevanceMode.SELECTIVE_ATTENTION:
            # Boost novel or surprising items
            if self._is_novel(item):
                modifier *= 1.5
                
        elif mode == RelevanceMode.WORKING_MEMORY:
            # Boost recently active items
            if self._is_recently_active(item):
                modifier *= 1.3
                
        elif mode == RelevanceMode.PROBLEM_SPACE:
            # Boost goal-relevant items
            if self._is_goal_relevant(item, context):
                modifier *= 2.0
                
        return modifier
        
    def _is_novel(self, item: Any) -> bool:
        """Check if an item is novel (not in recent salience history)."""
        recent_items = {record.context.get('item') for record in self.salience_history[-10:]}
        return item not in recent_items
        
    def _is_recently_active(self, item: Any) -> bool:
        """Check if an item was recently active."""
        return item in self.current_focus
        
    def _is_goal_relevant(self, item: Any, context: Dict) -> bool:
        """Check if an item is relevant to current goals."""
        goals = context.get('goals', [])
        if not goals:
            return False
            
        # Simple relevance check - could be enhanced with semantic similarity
        item_name = getattr(item, 'name', str(item))
        return any(goal.lower() in item_name.lower() for goal in goals)
        
    def _integrate_relevance_results(self, mode_results: Dict[str, Dict], 
                                   context: Dict) -> Dict[str, Any]:
        """Integrate relevance results across all modes using meta-cognitive processes."""
        all_relevant = set()
        combined_landscape = {}
        total_confidence = 0.0
        
        # Combine results from all modes
        for mode_name, result in mode_results.items():
            all_relevant.update(result['relevant_items'])
            
            # Weighted combination of salience landscapes
            mode_weight = self._get_mode_weight(mode_name, context)
            for item, salience in result['salience_landscape'].items():
                if item not in combined_landscape:
                    combined_landscape[item] = 0.0
                combined_landscape[item] += salience * mode_weight
                
            total_confidence += result['confidence']
            
        # Normalize confidence
        avg_confidence = total_confidence / len(mode_results)
        
        # Apply meta-cognitive filtering
        filtered_relevant = self._apply_meta_cognitive_filter(all_relevant, context)
        
        # Compute attention allocation
        attention_allocation = self._compute_attention_allocation(combined_landscape)
        
        return {
            'relevant_atoms': filtered_relevant,
            'salience_landscape': combined_landscape,
            'attention_allocation': attention_allocation,
            'confidence': avg_confidence
        }
        
    def _get_mode_weight(self, mode_name: str, context: Dict) -> float:
        """Get weight for a specific relevance mode based on context."""
        default_weights = {
            'selective_attention': 1.0,
            'working_memory': 0.8,
            'problem_space': 1.2,
            'side_effects': 0.6,
            'long_term_memory': 0.7
        }
        
        # Allow context to override weights
        context_weights = context.get('mode_weights', {})
        return context_weights.get(mode_name, default_weights.get(mode_name, 1.0))
        
    def _apply_meta_cognitive_filter(self, relevant_items: Set[Any], 
                                   context: Dict) -> Set[Any]:
        """Apply meta-cognitive filtering to refine relevance."""
        filtered = set()
        
        for item in relevant_items:
            # Check meta-cognitive criteria
            if self._passes_meta_cognitive_criteria(item, context):
                filtered.add(item)
                
        return filtered
        
    def _passes_meta_cognitive_criteria(self, item: Any, context: Dict) -> bool:
        """Check if an item passes meta-cognitive filtering criteria."""
        # Coherence check
        if not self._is_coherent_with_context(item, context):
            return False
            
        # Utility check
        if not self._has_sufficient_utility(item, context):
            return False
            
        # Novelty vs. familiarity balance
        if not self._has_appropriate_novelty_level(item, context):
            return False
            
        return True
        
    def _is_coherent_with_context(self, item: Any, context: Dict) -> bool:
        """Check if item is coherent with current context."""
        # Simplified coherence check - could be enhanced
        return True  # Placeholder
        
    def _has_sufficient_utility(self, item: Any, context: Dict) -> bool:
        """Check if item has sufficient utility."""
        # Simplified utility check - could be enhanced
        return True  # Placeholder
        
    def _has_appropriate_novelty_level(self, item: Any, context: Dict) -> bool:
        """Check if item has appropriate novelty level."""
        # Balance between novelty and familiarity
        is_novel = self._is_novel(item)
        novelty_preference = context.get('novelty_preference', 0.5)
        
        if novelty_preference > 0.7:
            return is_novel
        elif novelty_preference < 0.3:
            return not is_novel
        else:
            return True  # Accept both novel and familiar
            
    def _compute_attention_allocation(self, salience_landscape: Dict[Any, float]) -> Dict[Any, float]:
        """Compute attention allocation based on salience landscape."""
        if not salience_landscape:
            return {}
            
        # Normalize saliences to probabilities
        total_salience = sum(salience_landscape.values())
        if total_salience == 0:
            return {}
            
        attention_allocation = {
            item: salience / total_salience 
            for item, salience in salience_landscape.items()
        }
        
        return attention_allocation
        
    def _track_salience_dynamics(self, result: Dict[str, Any], context: Dict):
        """Track salience dynamics over time."""
        import time
        
        timestamp = time.time()
        
        for item, salience in result['salience_landscape'].items():
            record = SalienceRecord(
                timestamp=timestamp,
                salience_value=salience,
                context=context.copy(),
                mode=RelevanceMode.SELECTIVE_ATTENTION  # Could be more specific
            )
            self.salience_history.append(record)
            
        # Trim history to reasonable size
        if len(self.salience_history) > 1000:
            self.salience_history = self.salience_history[-1000:]
            
    def _detect_emergent_patterns(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect emergent patterns in relevance realization."""
        patterns = []
        
        # Pattern 1: Attention clustering
        high_attention_items = [
            item for item, attention in result['attention_allocation'].items()
            if attention > 0.2
        ]
        
        if len(high_attention_items) > 1:
            patterns.append({
                'type': 'attention_cluster',
                'items': high_attention_items,
                'strength': len(high_attention_items) / len(result['attention_allocation'])
            })
            
        # Pattern 2: Salience oscillation
        if len(self.salience_history) >= 10:
            recent_values = [r.salience_value for r in self.salience_history[-10:]]
            if self._detect_oscillation(recent_values):
                patterns.append({
                    'type': 'salience_oscillation',
                    'values': recent_values,
                    'strength': self._measure_oscillation_strength(recent_values)
                })
                
        return patterns
        
    def _detect_oscillation(self, values: List[float]) -> bool:
        """Detect oscillation pattern in values."""
        if len(values) < 6:
            return False
            
        # Simple oscillation detection
        differences = [values[i+1] - values[i] for i in range(len(values)-1)]
        sign_changes = sum(1 for i in range(len(differences)-1) 
                          if differences[i] * differences[i+1] < 0)
        
        return sign_changes >= 3  # At least 3 sign changes indicate oscillation
        
    def _measure_oscillation_strength(self, values: List[float]) -> float:
        """Measure the strength of oscillation."""
        if not values:
            return 0.0
            
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        
        return min(1.0, variance)  # Normalize to [0, 1]
        
    def _update_attention_focus(self, result: Dict[str, Any]):
        """Update current attention focus."""
        # Update focus based on attention allocation
        high_attention_threshold = 0.1
        new_focus = {
            item for item, attention in result['attention_allocation'].items()
            if attention > high_attention_threshold
        }
        
        self.current_focus = new_focus
        
    def _get_meta_cognitive_state(self) -> Dict[str, Any]:
        """Get current meta-cognitive state."""
        return {
            'attention_bank_state': self.attention_bank.get_state(),
            'current_focus_size': len(self.current_focus),
            'salience_history_length': len(self.salience_history),
            'emergent_behaviors': list(self.emergent_behaviors),
            'meta_patterns': dict(self.meta_patterns)
        }


class AttentionBank:
    """
    OpenCog-style attention bank for managing attention allocation.
    """
    
    def __init__(self, atomspace: AtomSpace, total_sti: float = 1000.0, total_lti: float = 1000.0):
        """Initialize attention bank.
        
        Args:
            atomspace: The AtomSpace to manage attention for
            total_sti: Total available short-term importance
            total_lti: Total available long-term importance
        """
        self.atomspace = atomspace
        self.total_sti = total_sti
        self.total_lti = total_lti
        self.allocated_sti = 0.0
        self.allocated_lti = 0.0
        
    def update_atom_attention(self, atom: Any, new_av: AttentionValue):
        """Update attention value for an atom, managing total attention budget."""
        if hasattr(atom, 'get_attention_value'):
            old_av = atom.get_attention_value()
            
            # Update allocated attention
            sti_diff = new_av.sti - old_av.sti
            lti_diff = new_av.lti - old_av.lti
            
            # Check if we have enough attention budget
            if self.allocated_sti + sti_diff <= self.total_sti:
                self.allocated_sti += sti_diff
                
                if self.allocated_lti + lti_diff <= self.total_lti:
                    self.allocated_lti += lti_diff
                    atom.set_attention_value(new_av)
                    return True
                    
        return False
        
    def normalize_attention(self):
        """Normalize attention values to fit within budget."""
        if self.allocated_sti > self.total_sti or self.allocated_lti > self.total_lti:
            # Need to redistribute attention
            all_atoms = [atom for atom in self.atomspace.get_all_atoms() 
                        if hasattr(atom, 'get_attention_value')]
            
            if all_atoms:
                sti_factor = self.total_sti / max(1.0, self.allocated_sti)
                lti_factor = self.total_lti / max(1.0, self.allocated_lti)
                
                for atom in all_atoms:
                    av = atom.get_attention_value()
                    new_av = AttentionValue(
                        av.sti * sti_factor,
                        av.lti * lti_factor,
                        av.vlti
                    )
                    atom.set_attention_value(new_av)
                    
                self.allocated_sti = self.total_sti
                self.allocated_lti = self.total_lti
                
    def get_state(self) -> Dict[str, Any]:
        """Get current attention bank state."""
        return {
            'total_sti': self.total_sti,
            'total_lti': self.total_lti,
            'allocated_sti': self.allocated_sti,
            'allocated_lti': self.allocated_lti,
            'sti_utilization': self.allocated_sti / self.total_sti,
            'lti_utilization': self.allocated_lti / self.total_lti
        }


class OpenCogPatternMatcher:
    """
    OpenCog-style pattern matcher for relevance realization.
    """
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize pattern matcher.
        
        Args:
            atomspace: The AtomSpace to perform pattern matching on
        """
        self.atomspace = atomspace
        
    def match_pattern(self, pattern: Any) -> List[Any]:
        """Match a pattern in the AtomSpace.
        
        Args:
            pattern: Pattern to match (can contain None as wildcards)
            
        Returns:
            List of matching atoms
        """
        # Simplified pattern matching - could be enhanced
        if hasattr(pattern, 'is_link') and pattern.is_link():
            return self._match_link_pattern(pattern)
        elif hasattr(pattern, 'is_node') and pattern.is_node():
            return self._match_node_pattern(pattern)
        else:
            return []
            
    def _match_link_pattern(self, pattern_link: Any) -> List[Any]:
        """Match a link pattern."""
        matches = []
        
        # Get all links of the same type
        all_links = self.atomspace.get_atoms_by_type(pattern_link.atom_type)
        
        for link in all_links:
            if hasattr(link, 'is_link') and link.is_link():
                if self._link_matches_pattern(link, pattern_link):
                    matches.append(link)
                    
        return matches
        
    def _match_node_pattern(self, pattern_node: Any) -> List[Any]:
        """Match a node pattern."""
        # Exact node matching
        if hasattr(pattern_node, 'name') and pattern_node.name:
            exact_match = self.atomspace.get_atom_by_type_name(
                pattern_node.atom_type, pattern_node.name
            )
            return [exact_match] if exact_match else []
        else:
            # Return all nodes of this type
            return self.atomspace.get_atoms_by_type(pattern_node.atom_type)
            
    def _link_matches_pattern(self, link: Any, pattern: Any) -> bool:
        """Check if a link matches a pattern."""
        if not (hasattr(link, 'outgoing_set') and hasattr(pattern, 'outgoing_set')):
            return False
            
        if len(link.outgoing_set) != len(pattern.outgoing_set):
            return False
            
        for i, (link_atom, pattern_atom) in enumerate(zip(link.outgoing_set, pattern.outgoing_set)):
            if pattern_atom is None:
                # Wildcard matches anything
                continue
            elif hasattr(pattern_atom, 'id') and hasattr(link_atom, 'id'):
                if link_atom.id != pattern_atom.id:
                    return False
            else:
                return False
                
        return True


class MeaningMaker:
    """
    Meaning-making system implementing Vervaeke's framework for relevance realization.
    """
    
    def __init__(self, atomspace: AtomSpace):
        """Initialize meaning maker.
        
        Args:
            atomspace: The AtomSpace for knowledge representation
        """
        self.atomspace = atomspace
        
    def extract_meaning(self, atoms: Set[Any], context: Dict) -> Dict[str, Any]:
        """Extract meaning from atoms using Vervaeke's three types of patterns.
        
        Args:
            atoms: Set of atoms to extract meaning from
            context: Current context
            
        Returns:
            Dictionary with meaning structures
        """
        meaning_structures = {
            'nomological': self._extract_nomological_patterns(atoms),
            'normative': self._extract_normative_patterns(atoms),
            'narrative': self._extract_narrative_patterns(atoms, context)
        }
        
        # Integrate patterns
        integrated_meaning = self._integrate_meaning_patterns(meaning_structures)
        
        return {
            'patterns': meaning_structures,
            'integrated_meaning': integrated_meaning,
            'coherence_score': self._compute_coherence_score(meaning_structures)
        }
        
    def _extract_nomological_patterns(self, atoms: Set[Any]) -> List[Dict[str, Any]]:
        """Extract nomological (causal/logical) patterns."""
        patterns = []
        
        # Look for causal links
        for atom in atoms:
            if hasattr(atom, 'incoming_set'):
                causal_links = [
                    link for link in atom.incoming_set 
                    if hasattr(link, 'atom_type') and 'causal' in link.atom_type.lower()
                ]
                
                if causal_links:
                    patterns.append({
                        'type': 'causal',
                        'atom': atom,
                        'links': causal_links
                    })
                    
        return patterns
        
    def _extract_normative_patterns(self, atoms: Set[Any]) -> List[Dict[str, Any]]:
        """Extract normative (evaluative) patterns."""
        patterns = []
        
        # Look for evaluation links
        for atom in atoms:
            if hasattr(atom, 'incoming_set'):
                eval_links = [
                    link for link in atom.incoming_set 
                    if hasattr(link, 'atom_type') and 'evaluation' in link.atom_type.lower()
                ]
                
                if eval_links:
                    patterns.append({
                        'type': 'evaluative',
                        'atom': atom,
                        'links': eval_links
                    })
                    
        return patterns
        
    def _extract_narrative_patterns(self, atoms: Set[Any], context: Dict) -> List[Dict[str, Any]]:
        """Extract narrative (sequential/temporal) patterns."""
        patterns = []
        
        # Look for temporal or sequential relationships
        temporal_atoms = []
        for atom in atoms:
            if hasattr(atom, 'name') and any(
                word in atom.name.lower() 
                for word in ['before', 'after', 'during', 'then', 'next']
            ):
                temporal_atoms.append(atom)
                
        if temporal_atoms:
            patterns.append({
                'type': 'temporal_sequence',
                'atoms': temporal_atoms
            })
            
        return patterns
        
    def _integrate_meaning_patterns(self, patterns: Dict[str, List]) -> Dict[str, Any]:
        """Integrate different types of meaning patterns."""
        integrated = {}
        
        # Count pattern types
        pattern_counts = {
            pattern_type: len(pattern_list) 
            for pattern_type, pattern_list in patterns.items()
        }
        
        # Determine dominant pattern type
        dominant_type = max(pattern_counts, key=pattern_counts.get) if pattern_counts else None
        
        integrated['dominant_pattern_type'] = dominant_type
        integrated['pattern_distribution'] = pattern_counts
        
        # Compute integration score
        total_patterns = sum(pattern_counts.values())
        if total_patterns > 0:
            integration_score = 1.0 - (max(pattern_counts.values()) / total_patterns)
        else:
            integration_score = 0.0
            
        integrated['integration_score'] = integration_score
        
        return integrated
        
    def _compute_coherence_score(self, patterns: Dict[str, List]) -> float:
        """Compute coherence score for meaning structures."""
        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
        
        if total_patterns == 0:
            return 0.0
            
        # Simple coherence metric - could be enhanced
        pattern_types = len([p for p in patterns.values() if p])
        coherence = pattern_types / 3.0  # Normalize by max pattern types
        
        return min(1.0, coherence) 