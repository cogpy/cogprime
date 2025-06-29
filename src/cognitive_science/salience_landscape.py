from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import numpy as np

from ..core.relevance_core import RelevanceCore, RelevanceMode

class SalienceLevel(Enum):
    """Different levels of salience processing"""
    FEATURIZATION = "featurization"  # Basic feature detection
    FOREGROUNDING = "foregrounding"  # Figure-ground separation
    FIGURATION = "figuration"        # Pattern configuration
    FRAMING = "framing"              # Problem/context framing

@dataclass
class SalienceVector:
    """Three-dimensional salience vector using ACT framework"""
    aspectuality: float  # How something is aspectualized/configured
    centrality: float   # How central/important to the agent
    temporality: float  # Temporal relevance/urgency

class SalienceLandscape:
    """Handles dynamic salience landscape based on ACT framework.
    
    Implements Vervaeke's model of salience as having three core dimensions:
    - Aspectuality: How things are configured/aspectualized
    - Centrality: How central/important to the agent
    - Temporality: Temporal relevance/kairos
    """
    
    def __init__(self, relevance_core: RelevanceCore):
        self.relevance_core = relevance_core
        self.current_landscape: Dict[str, SalienceVector] = {}
        self.processing_levels = {
            level: set() for level in SalienceLevel
        }
        
    def update_salience(self, item_id: str,
                       features: Dict,
                       context: Optional[Dict] = None) -> SalienceVector:
        """Update salience vector for an item based on features and context.
        
        Args:
            item_id: Identifier for the item
            features: Features of the item
            context: Optional contextual information
            
        Returns:
            Updated salience vector
        """
        # Get relevance evaluation
        relevant_features = self.relevance_core.evaluate_relevance(
            set(features.keys()),
            context=context
        )[0]
        
        # Compute aspectuality (how it's configured)
        aspectuality = self._compute_aspectuality(
            features, relevant_features
        )
        
        # Compute centrality (importance to agent)
        centrality = self._compute_centrality(
            features, relevant_features, context
        )
        
        # Compute temporality (temporal relevance)
        temporality = self._compute_temporality(
            features, relevant_features, context
        )
        
        # Create and store salience vector
        vector = SalienceVector(
            aspectuality=aspectuality,
            centrality=centrality,
            temporality=temporality
        )
        self.current_landscape[item_id] = vector
        
        return vector
        
    def process_level(self, level: SalienceLevel,
                     items: Set[str]) -> Set[str]:
        """Process items at a particular level of salience processing.
        
        Args:
            level: The processing level
            items: Set of item IDs to process
            
        Returns:
            Set of items that passed the processing
        """
        # Get current vectors
        vectors = {
            item: self.current_landscape[item]
            for item in items
            if item in self.current_landscape
        }
        
        # Apply level-specific processing
        if level == SalienceLevel.FEATURIZATION:
            return self._process_featurization(vectors)
        elif level == SalienceLevel.FOREGROUNDING:
            return self._process_foregrounding(vectors)
        elif level == SalienceLevel.FIGURATION:
            return self._process_figuration(vectors)
        else:  # FRAMING
            return self._process_framing(vectors)
            
    def _compute_aspectuality(self, features: Dict,
                            relevant_features: Set) -> float:
        """Compute aspectuality based on feature configuration."""
        # How well features cohere into an aspect
        feature_coherence = len(relevant_features) / len(features)
        
        # How distinctive the configuration is
        distinctiveness = np.mean([
            features.get(f, 0) for f in relevant_features
        ])
        
        return (feature_coherence + distinctiveness) / 2.0
        
    def _compute_centrality(self, features: Dict,
                          relevant_features: Set,
                          context: Optional[Dict]) -> float:
        """Compute centrality based on importance to agent."""
        # Base importance from relevance core
        importance = len(relevant_features) / len(features)
        
        # Adjust based on context if available
        if context and "agent_needs" in context:
            needs_alignment = np.mean([
                features.get(need, 0)
                for need in context["agent_needs"]
            ])
            importance = (importance + needs_alignment) / 2.0
            
        return importance
        
    def _compute_temporality(self, features: Dict,
                           relevant_features: Set,
                           context: Optional[Dict]) -> float:
        """Compute temporal relevance/urgency."""
        # Base temporal relevance
        temporal_relevance = 0.5  # Neutral by default
        
        # Adjust based on context if available
        if context and "temporal_horizon" in context:
            horizon = context["temporal_horizon"]
            temporal_features = {
                f for f in relevant_features
                if "time" in str(f) or "urgent" in str(f)
            }
            temporal_relevance = len(temporal_features) / (horizon + 1)
            
        return temporal_relevance
        
    def _process_featurization(self, vectors: Dict[str, SalienceVector]) -> Set[str]:
        """Process basic feature detection."""
        return {
            item for item, vector in vectors.items()
            if vector.aspectuality > 0.3  # Basic feature threshold
        }
        
    def _process_foregrounding(self, vectors: Dict[str, SalienceVector]) -> Set[str]:
        """Process figure-ground separation."""
        # Items with high enough centrality become foreground
        return {
            item for item, vector in vectors.items()
            if vector.centrality > 0.5
        }
        
    def _process_figuration(self, vectors: Dict[str, SalienceVector]) -> Set[str]:
        """Process pattern configuration."""
        # Items that form coherent patterns
        return {
            item for item, vector in vectors.items()
            if vector.aspectuality > 0.6 and vector.centrality > 0.4
        }
        
    def _process_framing(self, vectors: Dict[str, SalienceVector]) -> Set[str]:
        """Process problem/context framing."""
        # Items relevant for current frame
        return {
            item for item, vector in vectors.items()
            if (vector.aspectuality + vector.centrality + 
                vector.temporality) / 3.0 > 0.7
        } 