from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import numpy as np

from ..core.relevance_core import RelevanceCore, RelevanceMode

class CommunicationMaxim(Enum):
    """Grice's maxims as described by Vervaeke"""
    QUALITY = "quality"  # Be truthful/sincere
    QUANTITY = "quantity"  # Right amount of info
    MANNER = "manner"  # Clear presentation
    RELEVANCE = "relevance"  # Be relevant

@dataclass
class MeaningContext:
    """Context for meaning-making"""
    nomological: Dict  # Causal/scientific relations
    normative: Dict  # Value/ethical relations  
    narrative: Dict  # Story/temporal relations
    participatory: Dict  # Embodied/interactive relations

class MeaningMaker:
    """Handles meaning-making and communication based on relevance realization.
    
    This implements Vervaeke's framework for meaning cultivation through
    the integration of multiple ways of knowing and relating.
    """
    
    def __init__(self, relevance_core: RelevanceCore):
        self.relevance_core = relevance_core
        self.current_context = MeaningContext(
            nomological={},
            normative={},
            narrative={},
            participatory={}
        )
        
    def communicate(self, message: str, context: Dict) -> Tuple[str, float]:
        """Generate communication following Grice's maxims.
        
        Args:
            message: Core message to communicate
            context: Communication context
            
        Returns:
            Tuple of (refined message, confidence)
        """
        # Check maxims using relevance core
        maxim_scores = {}
        for maxim in CommunicationMaxim:
            relevant_aspects = self.relevance_core.evaluate_relevance(
                {message},
                context={
                    "maxim": maxim,
                    "context": context
                }
            )
            maxim_scores[maxim] = relevant_aspects[1]  # Get confidence
            
        # Refine message based on maxim scores
        refined_message = self._refine_message(
            message, maxim_scores, context
        )
        
        # Calculate overall confidence
        confidence = np.mean(list(maxim_scores.values()))
        
        return refined_message, confidence
        
    def cultivate_meaning(self, experience: Dict,
                         context: Optional[Dict] = None) -> MeaningContext:
        """Cultivate meaning from experience across multiple domains.
        
        Args:
            experience: Features of the experience
            context: Optional additional context
            
        Returns:
            Updated meaning context
        """
        # Update each aspect of meaning
        self._update_nomological(experience, context)
        self._update_normative(experience, context)
        self._update_narrative(experience, context)
        self._update_participatory(experience, context)
        
        return self.current_context
        
    def _refine_message(self, message: str,
                       maxim_scores: Dict[CommunicationMaxim, float],
                       context: Dict) -> str:
        """Refine message based on maxim scores."""
        # Placeholder for more sophisticated message refinement
        if maxim_scores[CommunicationMaxim.QUANTITY] < 0.5:
            message += " [More detail needed]"
        if maxim_scores[CommunicationMaxim.MANNER] < 0.5:
            message += " [Clarity needed]"
        return message
        
    def _update_nomological(self, experience: Dict,
                           context: Optional[Dict]) -> None:
        """Update causal/scientific relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "nomological"}
        )[0]
        
        self.current_context.nomological.update({
            k: experience[k] for k in relevant_features
            if k in experience
        })
        
    def _update_normative(self, experience: Dict,
                         context: Optional[Dict]) -> None:
        """Update value/ethical relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "normative"}
        )[0]
        
        self.current_context.normative.update({
            k: experience[k] for k in relevant_features
            if k in experience
        })
        
    def _update_narrative(self, experience: Dict,
                         context: Optional[Dict]) -> None:
        """Update story/temporal relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "narrative"}
        )[0]
        
        self.current_context.narrative.update({
            k: experience[k] for k in relevant_features
            if k in experience
        })
        
    def _update_participatory(self, experience: Dict,
                            context: Optional[Dict]) -> None:
        """Update embodied/interactive relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "participatory"}
        )[0]
        
        self.current_context.participatory.update({
            k: experience[k] for k in relevant_features
            if k in experience
        }) 