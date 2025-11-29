"""
Enhanced Perception Module with 4E Cognition Integration

Integrates Vervaeke's 4E framework with the existing perception module to enable:
- Embodied sensorimotor grounding
- Embedded contextual awareness
- Enacted active perception
- Extended cognitive coupling
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from ..modules.perception import SensoryInput, PerceptionModule
from ..cognitive_science.vervaeke_4e_framework import (
    EmbodiedCognitionProcessor,
    EmbeddedCognitionProcessor,
    EnactedCognitionProcessor,
    ExtendedCognitionProcessor
)


@dataclass
class EnhancedSensoryInput:
    """Extended sensory input including motor and emotional states"""
    visual: Optional[torch.Tensor] = None
    auditory: Optional[torch.Tensor] = None
    proprioceptive: Optional[torch.Tensor] = None
    motor_state: Optional[torch.Tensor] = None
    emotional_state: Optional[torch.Tensor] = None
    text: Optional[str] = None
    context_info: Optional[Dict] = None


class Embodied4EPerceptionModule(nn.Module):
    """
    Enhanced perception module with full 4E cognition integration.
    
    Combines traditional perception with embodied, embedded, enacted, and
    extended cognition principles for richer sensorimotor understanding.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Base perception module
        self.perception = PerceptionModule(config)
        
        feature_dim = config.get('feature_dim', 512)
        motor_dim = config.get('motor_dim', 128)
        
        # 4E cognition processors
        self.embodied_processor = EmbodiedCognitionProcessor(
            sensory_dim=feature_dim,
            motor_dim=motor_dim
        )
        
        self.embedded_processor = EmbeddedCognitionProcessor(
            feature_dim=feature_dim,
            context_dim=config.get('context_dim', 256)
        )
        
        self.enacted_processor = EnactedCognitionProcessor(
            feature_dim=feature_dim,
            action_dim=config.get('action_dim', 64)
        )
        
        self.extended_processor = ExtendedCognitionProcessor(
            feature_dim=feature_dim,
            tool_dim=config.get('tool_dim', 128)
        )
        
        # Fusion network to integrate all 4E aspects
        fusion_input_dim = feature_dim * 2  # Base + embodied
        self.four_e_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, 
                enhanced_input: EnhancedSensoryInput,
                context: Optional[torch.Tensor] = None,
                action: Optional[torch.Tensor] = None,
                tool_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process enhanced sensory input through 4E cognition pipeline.
        
        Args:
            enhanced_input: Sensory input with motor and emotional states
            context: Optional context tensor
            action: Optional action for enacted processing
            tool_state: Optional tool state for extended processing
            
        Returns:
            four_e_perception: Integrated 4E perceptual state
            comprehensive_metrics: All perception and 4E metrics
        """
        all_metrics = {}
        
        # 1. Base perception processing
        base_sensory = SensoryInput(
            visual=enhanced_input.visual,
            auditory=enhanced_input.auditory,
            proprioceptive=enhanced_input.proprioceptive,
            text=enhanced_input.text
        )
        
        base_perception, perception_metrics = self.perception.process_input(
            base_sensory, context
        )
        all_metrics['perception'] = perception_metrics
        
        # 2. Embodied cognition - integrate with motor state
        if enhanced_input.motor_state is not None:
            embodied_state, embodied_metrics = self.embodied_processor(
                base_perception,
                enhanced_input.motor_state
            )
            all_metrics['embodied'] = embodied_metrics
        else:
            # Use zero motor state if not provided
            motor_state = torch.zeros(self.embodied_processor.motor_dim)
            embodied_state, embodied_metrics = self.embodied_processor(
                base_perception,
                motor_state
            )
            all_metrics['embodied'] = embodied_metrics
        
        # 3. Embedded cognition - contextualize perception
        embedded_state, embedded_metrics = self.embedded_processor(
            embodied_state,
            enhanced_input.context_info
        )
        all_metrics['embedded'] = embedded_metrics
        
        # 4. Enacted cognition - active perception
        enacted_state, enacted_metrics = self.enacted_processor(
            embedded_state,
            action
        )
        all_metrics['enacted'] = enacted_metrics
        
        # 5. Extended cognition - tool-mediated perception
        extended_state, extended_metrics = self.extended_processor(
            enacted_state,
            tool_state
        )
        all_metrics['extended'] = extended_metrics
        
        # 6. Fuse all 4E aspects with base perception
        combined = torch.cat([base_perception, extended_state], dim=-1)
        four_e_perception = self.four_e_fusion(combined)
        
        # 7. Compute integrated metrics
        all_metrics['four_e_integration'] = self._compute_integration_metrics(
            all_metrics
        )
        
        return four_e_perception, all_metrics
    
    def _compute_integration_metrics(self, all_metrics: Dict) -> Dict[str, float]:
        """Compute metrics for 4E integration quality"""
        
        # Sensorimotor grounding strength
        sensorimotor_grounding = all_metrics['embodied']['sensorimotor_coupling']
        
        # Contextual adaptation
        contextual_adaptation = all_metrics['embedded']['environmental_coupling']
        
        # Active exploration engagement
        exploration_engagement = all_metrics['enacted']['exploration_intensity']
        
        # Cognitive extension utilization
        extension_utilization = all_metrics['extended']['cognitive_extension']
        
        # Overall 4E integration
        overall_integration = (
            sensorimotor_grounding * 0.3 +
            contextual_adaptation * 0.3 +
            exploration_engagement * 0.2 +
            extension_utilization * 0.2
        )
        
        return {
            'sensorimotor_grounding': sensorimotor_grounding,
            'contextual_adaptation': contextual_adaptation,
            'exploration_engagement': exploration_engagement,
            'extension_utilization': extension_utilization,
            'overall_4e_integration': overall_integration
        }
    
    def update_body_schema(self, feedback: Dict[str, float]):
        """Update embodied processor's body schema"""
        self.embodied_processor.update_body_schema(feedback)
    
    def register_tool(self, tool_name: str, tool_features: torch.Tensor):
        """Register a cognitive tool in extended processor"""
        self.extended_processor.register_tool(tool_name, tool_features)
    
    def use_tool(self, tool_name: str) -> Optional[torch.Tensor]:
        """Use a registered cognitive tool"""
        return self.extended_processor.use_tool(tool_name)


class SalienceLandscapeNavigator(nn.Module):
    """
    Integrates salience landscape with 4E perception for attention guidance.
    
    Uses 4E cognition to modulate salience detection and navigate the
    attention landscape for action selection.
    """
    
    def __init__(self, feature_dim: int = 512, salience_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.salience_dim = salience_dim
        
        # Salience field generator
        self.salience_generator = nn.Sequential(
            nn.Linear(feature_dim, salience_dim),
            nn.ReLU(),
            nn.Linear(salience_dim, salience_dim),
            nn.Sigmoid()
        )
        
        # Relevance gradient computer
        self.relevance_gradient = nn.Sequential(
            nn.Linear(salience_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Attention peak detector
        self.peak_detector = nn.Sequential(
            nn.Linear(salience_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.salience_history = []
        
    def forward(self, 
                four_e_perception: torch.Tensor,
                four_e_metrics: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Navigate salience landscape using 4E perception.
        
        Args:
            four_e_perception: Integrated 4E perceptual state
            four_e_metrics: Metrics from 4E processing
            
        Returns:
            attention_target: Where to direct attention
            navigation_metrics: Salience navigation metrics
        """
        # Generate salience field
        salience_field = self.salience_generator(four_e_perception)
        
        # Modulate by 4E integration quality
        integration_quality = four_e_metrics.get(
            'overall_4e_integration', 0.5
        )
        modulated_salience = salience_field * (0.5 + integration_quality)
        
        # Compute relevance gradient
        relevance_direction = self.relevance_gradient(modulated_salience)
        
        # Detect attention peaks
        peak_strength = self.peak_detector(modulated_salience)
        
        # Track salience over time using deque for efficiency
        if not hasattr(self, '_salience_deque'):
            self._salience_deque = deque(maxlen=20)
        self._salience_deque.append(modulated_salience.detach())
        self.salience_history = list(self._salience_deque)
        
        # Compute navigation metrics
        landscape_coherence = self._compute_landscape_coherence()
        
        metrics = {
            'peak_strength': float(peak_strength),
            'landscape_coherence': landscape_coherence,
            'salience_intensity': float(torch.mean(modulated_salience)),
            'relevance_magnitude': float(torch.norm(relevance_direction))
        }
        
        # Attention target is relevance direction weighted by peak strength
        attention_target = relevance_direction * peak_strength
        
        return attention_target, metrics
    
    def _compute_landscape_coherence(self) -> float:
        """Compute coherence of salience landscape over time"""
        if len(self.salience_history) < 2:
            return 1.0
        
        # Measure stability of salience patterns
        salience_stack = torch.stack(self.salience_history)
        temporal_variance = float(torch.var(salience_stack, dim=0).mean())
        
        # Lower variance = higher coherence
        coherence = 1.0 / (1.0 + temporal_variance)
        
        return coherence


class Embodied4EPerceptionSystem:
    """
    Complete 4E perception system integrating all components.
    
    Provides a unified interface for embodied, embedded, enacted, and extended
    perception with salience landscape navigation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core 4E perception module
        self.perception_4e = Embodied4EPerceptionModule(config)
        
        # Salience landscape navigator
        self.salience_navigator = SalienceLandscapeNavigator(
            feature_dim=config.get('feature_dim', 512),
            salience_dim=config.get('salience_dim', 256)
        )
        
    def process_and_navigate(self,
                            enhanced_input: EnhancedSensoryInput,
                            context: Optional[torch.Tensor] = None,
                            action: Optional[torch.Tensor] = None,
                            tool_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Complete 4E perception and salience navigation cycle.
        
        Args:
            enhanced_input: Sensory input with motor/emotional states
            context: Optional context tensor
            action: Optional action for enaction
            tool_state: Optional tool state
            
        Returns:
            perception_state: Integrated 4E perception
            attention_target: Where to direct attention
            all_metrics: Complete metrics from all processing
        """
        # Process through 4E perception
        perception_state, perception_metrics = self.perception_4e(
            enhanced_input,
            context,
            action,
            tool_state
        )
        
        # Navigate salience landscape
        attention_target, navigation_metrics = self.salience_navigator(
            perception_state,
            perception_metrics['four_e_integration']
        )
        
        # Combine all metrics
        all_metrics = {
            **perception_metrics,
            'salience_navigation': navigation_metrics
        }
        
        return perception_state, attention_target, all_metrics
    
    def update_body_schema(self, feedback: Dict[str, float]):
        """Update body schema from sensory feedback"""
        self.perception_4e.update_body_schema(feedback)
    
    def register_tool(self, tool_name: str, tool_features: torch.Tensor):
        """Register a cognitive tool"""
        self.perception_4e.register_tool(tool_name, tool_features)
    
    def use_tool(self, tool_name: str) -> Optional[torch.Tensor]:
        """Use a registered tool"""
        return self.perception_4e.use_tool(tool_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'body_schema_joints': len(self.perception_4e.embodied_processor.body_schema.joint_states),
            'registered_tools': len(self.perception_4e.extended_processor.tool_mastery),
            'action_history_size': len(self.perception_4e.enacted_processor.action_history),
            'salience_history_size': len(self.salience_navigator.salience_history)
        }
