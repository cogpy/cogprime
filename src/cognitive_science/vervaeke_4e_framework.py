"""
Vervaeke 4E Cognition Framework Implementation

Integrates embodied, embedded, enacted, and extended cognition principles
following John Vervaeke's framework for addressing the meaning crisis.

This module provides computational implementations of:
- Embodied cognition: Sensorimotor grounding and body schema
- Embedded cognition: Environmental context and affordances
- Enacted cognition: Active perception and exploration
- Extended cognition: Tool use and cognitive distribution
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ProcessingMode(Enum):
    """Processing modes for cognitive adaptation"""
    CONFORMATIVE = "conformative"  # Assimilation to existing schemas
    TRANSFORMATIVE = "transformative"  # Accommodation and insight


@dataclass
class BodySchema:
    """Representation of body state and sensorimotor mappings"""
    joint_states: Dict[str, float] = field(default_factory=dict)
    sensorimotor_map: torch.Tensor = None
    body_boundary: float = 1.0
    spatial_extent: float = 1.0
    
    def update_from_feedback(self, feedback: Dict[str, float]):
        """Update body schema from sensory feedback"""
        for key, value in feedback.items():
            self.joint_states[key] = value


@dataclass
class EnvironmentalContext:
    """Physical and social environmental context"""
    physical_features: Dict[str, Any] = field(default_factory=dict)
    affordances: List[str] = field(default_factory=list)
    social_agents: List[str] = field(default_factory=list)
    temporal_dynamics: Dict[str, float] = field(default_factory=dict)
    stability: float = 1.0


@dataclass
class ActionOutcome:
    """Records action-outcome pairs for enacted learning"""
    action: str
    sensory_prediction: torch.Tensor
    actual_outcome: torch.Tensor
    prediction_error: float
    timestamp: float


class EmbodiedCognitionProcessor(nn.Module):
    """
    Processes embodied cognition - sensorimotor integration with body schema.
    
    Implements the principle that cognition is grounded in bodily experience
    and sensorimotor contingencies.
    """
    
    def __init__(self, sensory_dim: int = 512, motor_dim: int = 128):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        
        # Sensorimotor integration network
        self.sensorimotor_integrator = nn.Sequential(
            nn.Linear(sensory_dim + motor_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, sensory_dim)
        )
        
        # Body schema representation
        self.body_schema = BodySchema()
        self.body_schema.sensorimotor_map = torch.eye(min(sensory_dim, motor_dim))
        
    def forward(self, sensory_input: torch.Tensor, motor_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process embodied cognition with sensorimotor integration.
        
        Args:
            sensory_input: Current sensory state
            motor_state: Current motor configuration
            
        Returns:
            integrated_state: Embodied cognitive state
            metrics: Embodiment metrics
        """
        # Concatenate sensory and motor states
        combined = torch.cat([sensory_input, motor_state], dim=-1)
        
        # Process through sensorimotor integration
        integrated = self.sensorimotor_integrator(combined)
        
        # Compute embodiment metrics
        coupling_strength = float(torch.cosine_similarity(
            sensory_input.unsqueeze(0), 
            integrated.unsqueeze(0)
        ))
        
        spatial_awareness = float(torch.norm(motor_state)) / (self.motor_dim ** 0.5)
        
        metrics = {
            'sensorimotor_coupling': coupling_strength,
            'spatial_awareness': spatial_awareness,
            'body_boundary': self.body_schema.body_boundary
        }
        
        return integrated, metrics
    
    def update_body_schema(self, feedback: Dict[str, float]):
        """Update body schema from sensory feedback"""
        self.body_schema.update_from_feedback(feedback)


class EmbeddedCognitionProcessor(nn.Module):
    """
    Processes embedded cognition - environmental context and affordances.
    
    Implements the principle that cognition is situated in and shaped by
    environmental context.
    """
    
    def __init__(self, feature_dim: int = 512, context_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # Affordance detector
        self.affordance_detector = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        
        self.environmental_context = EnvironmentalContext()
        
    def forward(self, features: torch.Tensor, context_info: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process embedded cognition with context awareness.
        
        Args:
            features: Input features
            context_info: Optional environmental context information
            
        Returns:
            contextualized_features: Context-aware cognitive state
            metrics: Embedding metrics
        """
        # Encode context
        context_embedding = self.context_encoder(features)
        
        # Detect affordances
        affordances = self.affordance_detector(context_embedding)
        
        # Update environmental context
        if context_info:
            self.environmental_context.physical_features.update(context_info)
        
        # Compute embedding metrics
        coupling = float(torch.mean(affordances))
        stability = self.environmental_context.stability
        
        metrics = {
            'environmental_coupling': coupling,
            'affordance_detection': float(torch.max(affordances)),
            'context_stability': stability
        }
        
        return context_embedding, metrics
    
    def detect_affordances(self, context_embedding: torch.Tensor) -> List[str]:
        """Detect action affordances in current context"""
        affordance_vector = self.affordance_detector(context_embedding)
        # Simple threshold-based detection
        threshold = 0.5
        detected = []
        for i, val in enumerate(affordance_vector.squeeze()):
            if val > threshold:
                detected.append(f"affordance_{i}")
        return detected


class EnactedCognitionProcessor(nn.Module):
    """
    Processes enacted cognition - active perception and exploration.
    
    Implements the principle that cognition is brought forth through
    action and interaction with the world.
    """
    
    def __init__(self, feature_dim: int = 512, action_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # Action-perception predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Exploration strategy network
        self.exploration_strategy = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.action_history: List[ActionOutcome] = []
        
    def forward(self, features: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process enacted cognition with active perception.
        
        Args:
            features: Current perceptual state
            action: Optional action to predict outcome for
            
        Returns:
            enacted_state: Perception shaped by action
            metrics: Enaction metrics
        """
        if action is None:
            # Generate exploratory action
            action = self.exploration_strategy(features)
        
        # Predict sensory outcome of action
        combined = torch.cat([features, action], dim=-1)
        predicted_state = self.action_predictor(combined)
        
        # Compute enaction metrics
        exploration_intensity = float(torch.norm(action)) / (self.action_dim ** 0.5)
        prediction_confidence = float(torch.norm(predicted_state)) / (self.feature_dim ** 0.5)
        
        metrics = {
            'exploration_intensity': exploration_intensity,
            'prediction_confidence': prediction_confidence,
            'action_history_size': len(self.action_history)
        }
        
        return predicted_state, metrics
    
    def learn_action_outcome(self, action: torch.Tensor, predicted: torch.Tensor, actual: torch.Tensor, timestamp: float):
        """Learn from action-outcome experience"""
        error = float(torch.mean((predicted - actual) ** 2))
        outcome = ActionOutcome(
            action=str(action.shape),
            sensory_prediction=predicted,
            actual_outcome=actual,
            prediction_error=error,
            timestamp=timestamp
        )
        self.action_history.append(outcome)
        
        # Keep only recent history
        if len(self.action_history) > 100:
            self.action_history.pop(0)


class ExtendedCognitionProcessor(nn.Module):
    """
    Processes extended cognition - tool use and environmental coupling.
    
    Implements the principle that cognition extends beyond the brain to
    include tools, symbols, and environmental structures.
    """
    
    def __init__(self, feature_dim: int = 512, tool_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.tool_dim = tool_dim
        
        # Tool integration network
        self.tool_integrator = nn.Sequential(
            nn.Linear(feature_dim + tool_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Tool mastery tracker
        self.tool_mastery = {}
        self.distributed_memory = {}
        
    def forward(self, cognitive_state: torch.Tensor, tool_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process extended cognition with tool integration.
        
        Args:
            cognitive_state: Internal cognitive state
            tool_state: Optional tool/environmental extension state
            
        Returns:
            extended_state: Cognitively extended state
            metrics: Extension metrics
        """
        if tool_state is None:
            tool_state = torch.zeros(self.tool_dim)
        
        # Integrate tool into cognitive processing
        combined = torch.cat([cognitive_state, tool_state], dim=-1)
        extended = self.tool_integrator(combined)
        
        # Compute extension metrics
        tool_usage = float(torch.norm(tool_state)) / (self.tool_dim ** 0.5)
        extension_degree = float(torch.cosine_similarity(
            cognitive_state.unsqueeze(0),
            extended.unsqueeze(0)
        ))
        
        metrics = {
            'tool_usage': tool_usage,
            'cognitive_extension': extension_degree,
            'tool_repertoire': len(self.tool_mastery)
        }
        
        return extended, metrics
    
    def register_tool(self, tool_name: str, tool_features: torch.Tensor):
        """Register a new cognitive tool"""
        self.tool_mastery[tool_name] = {
            'features': tool_features,
            'usage_count': 0,
            'mastery_level': 0.0
        }
    
    def use_tool(self, tool_name: str) -> Optional[torch.Tensor]:
        """Use a registered tool"""
        if tool_name in self.tool_mastery:
            self.tool_mastery[tool_name]['usage_count'] += 1
            self.tool_mastery[tool_name]['mastery_level'] = min(
                1.0,
                self.tool_mastery[tool_name]['mastery_level'] + 0.01
            )
            return self.tool_mastery[tool_name]['features']
        return None


class PerspectivalParticipatoryProcessor(nn.Module):
    """
    Processes perspectival and participatory knowing.
    
    Implements perspectival knowing (knowing what it's like from a viewpoint)
    and participatory knowing (knowing through identity and belonging).
    """
    
    def __init__(self, feature_dim: int = 512, perspective_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.perspective_dim = perspective_dim
        
        # Perspective encoder
        self.perspective_encoder = nn.Sequential(
            nn.Linear(feature_dim, perspective_dim),
            nn.ReLU(),
            nn.Linear(perspective_dim, perspective_dim)
        )
        
        # Multiple viewpoint integrator
        self.viewpoint_integrator = nn.Sequential(
            nn.Linear(perspective_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, perspective_dim)
        )
        
        self.active_perspectives: List[torch.Tensor] = []
        
    def forward(self, features: torch.Tensor, context_perspective: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process perspectival and participatory knowing.
        
        Args:
            features: Input features to be perspectivized
            context_perspective: Optional contextual perspective
            
        Returns:
            perspectival_state: Perspective-aware state
            metrics: Perspectival metrics
        """
        # Encode current perspective
        perspective = self.perspective_encoder(features)
        
        if context_perspective is not None:
            # Integrate multiple perspectives
            combined = torch.cat([perspective, context_perspective], dim=-1)
            integrated = self.viewpoint_integrator(combined)
        else:
            integrated = perspective
        
        # Track active perspectives
        self.active_perspectives.append(perspective.detach())
        if len(self.active_perspectives) > 10:
            self.active_perspectives.pop(0)
        
        # Compute metrics
        flexibility = self._compute_perspective_flexibility()
        participation = self._compute_participation_depth(integrated)
        
        metrics = {
            'perspective_flexibility': flexibility,
            'participation_depth': participation,
            'active_viewpoints': len(self.active_perspectives)
        }
        
        return integrated, metrics
    
    def _compute_perspective_flexibility(self) -> float:
        """Compute flexibility in perspective shifting"""
        if len(self.active_perspectives) < 2:
            return 0.0
        
        # Measure diversity of perspectives
        perspectives = torch.stack(self.active_perspectives)
        pairwise_dist = torch.cdist(perspectives, perspectives)
        return float(torch.mean(pairwise_dist))
    
    def _compute_participation_depth(self, state: torch.Tensor) -> float:
        """Compute depth of participatory engagement"""
        # Simple measure based on state activation
        return float(torch.mean(torch.abs(state)))


class ProcessingModeManager:
    """
    Manages conformative vs transformative processing modes.
    
    Conformative: Assimilate new information into existing schemas
    Transformative: Transform schemas through accommodation and insight
    """
    
    def __init__(self):
        self.current_mode = ProcessingMode.CONFORMATIVE
        self.mode_history: List[ProcessingMode] = []
        self.transformation_threshold = 0.7
        
    def select_mode(self, novelty: float, coherence: float) -> ProcessingMode:
        """
        Select processing mode based on novelty and coherence.
        
        High novelty + low coherence -> Transformative
        Low novelty + high coherence -> Conformative
        """
        if novelty > self.transformation_threshold and coherence < 0.5:
            mode = ProcessingMode.TRANSFORMATIVE
        else:
            mode = ProcessingMode.CONFORMATIVE
        
        self.current_mode = mode
        self.mode_history.append(mode)
        
        if len(self.mode_history) > 50:
            self.mode_history.pop(0)
        
        return mode
    
    def get_mode_statistics(self) -> Dict[str, float]:
        """Get statistics on mode usage"""
        if not self.mode_history:
            return {'conformative_ratio': 0.5, 'transformative_ratio': 0.5}
        
        conformative_count = sum(1 for m in self.mode_history if m == ProcessingMode.CONFORMATIVE)
        transformative_count = len(self.mode_history) - conformative_count
        
        return {
            'conformative_ratio': conformative_count / len(self.mode_history),
            'transformative_ratio': transformative_count / len(self.mode_history)
        }


class CognitiveEmotionalIntegrator(nn.Module):
    """
    Integrates cognitive and emotional processing following Vervaeke's framework.
    
    Emotion and cognition are not separate but integrated in meaning-making
    and relevance realization.
    """
    
    def __init__(self, cognitive_dim: int = 512, emotion_dim: int = 64):
        super().__init__()
        self.cognitive_dim = cognitive_dim
        self.emotion_dim = emotion_dim
        
        # Cognitive-emotional fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(cognitive_dim + emotion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, cognitive_dim)
        )
        
        # Meaning coherence evaluator
        self.coherence_evaluator = nn.Sequential(
            nn.Linear(cognitive_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, cognitive_state: torch.Tensor, emotional_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Integrate cognitive and emotional processing.
        
        Args:
            cognitive_state: Current cognitive state
            emotional_state: Current emotional state
            
        Returns:
            integrated_state: Cognitively-emotionally integrated state
            metrics: Integration metrics
        """
        # Fuse cognitive and emotional states
        combined = torch.cat([cognitive_state, emotional_state], dim=-1)
        integrated = self.fusion_network(combined)
        
        # Evaluate meaning coherence
        coherence = self.coherence_evaluator(integrated)
        
        # Compute synchrony between cognitive and emotional
        synchrony = float(torch.cosine_similarity(
            cognitive_state.unsqueeze(0),
            integrated.unsqueeze(0)
        ))
        
        metrics = {
            'cognitive_emotional_synchrony': synchrony,
            'meaning_coherence': float(coherence),
            'integration_quality': synchrony * float(coherence)
        }
        
        return integrated, metrics


class Vervaeke4EFramework(nn.Module):
    """
    Complete integration of Vervaeke's 4E cognition framework.
    
    Coordinates embodied, embedded, enacted, and extended cognition with
    perspectival/participatory knowing and cognitive-emotional integration.
    """
    
    def __init__(self, config: Dict[str, int]):
        super().__init__()
        
        feature_dim = config.get('feature_dim', 512)
        
        # Initialize all processors
        self.embodied = EmbodiedCognitionProcessor(
            sensory_dim=feature_dim,
            motor_dim=config.get('motor_dim', 128)
        )
        
        self.embedded = EmbeddedCognitionProcessor(
            feature_dim=feature_dim,
            context_dim=config.get('context_dim', 256)
        )
        
        self.enacted = EnactedCognitionProcessor(
            feature_dim=feature_dim,
            action_dim=config.get('action_dim', 64)
        )
        
        self.extended = ExtendedCognitionProcessor(
            feature_dim=feature_dim,
            tool_dim=config.get('tool_dim', 128)
        )
        
        self.perspectival_participatory = PerspectivalParticipatoryProcessor(
            feature_dim=feature_dim,
            perspective_dim=config.get('perspective_dim', 128)
        )
        
        self.cognitive_emotional = CognitiveEmotionalIntegrator(
            cognitive_dim=feature_dim,
            emotion_dim=config.get('emotion_dim', 64)
        )
        
        self.mode_manager = ProcessingModeManager()
        
    def process_cycle(self,
                     sensory_input: torch.Tensor,
                     motor_state: torch.Tensor,
                     emotional_state: torch.Tensor,
                     context_info: Optional[Dict] = None,
                     action: Optional[torch.Tensor] = None,
                     tool_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute a complete 4E cognitive cycle.
        
        Args:
            sensory_input: Sensory information
            motor_state: Motor/body state
            emotional_state: Emotional state
            context_info: Environmental context
            action: Optional action for enaction
            tool_state: Optional tool state for extension
            
        Returns:
            final_state: Integrated cognitive state
            all_metrics: Comprehensive metrics from all processors
        """
        all_metrics = {}
        
        # 1. Embodied processing
        embodied_state, embodied_metrics = self.embodied(sensory_input, motor_state)
        all_metrics['embodied'] = embodied_metrics
        
        # 2. Embedded processing
        embedded_state, embedded_metrics = self.embedded(embodied_state, context_info)
        all_metrics['embedded'] = embedded_metrics
        
        # 3. Enacted processing
        enacted_state, enacted_metrics = self.enacted(embedded_state, action)
        all_metrics['enacted'] = enacted_metrics
        
        # 4. Extended processing
        extended_state, extended_metrics = self.extended(enacted_state, tool_state)
        all_metrics['extended'] = extended_metrics
        
        # 5. Perspectival/Participatory processing
        perspectival_state, perspectival_metrics = self.perspectival_participatory(extended_state)
        all_metrics['perspectival'] = perspectival_metrics
        
        # 6. Cognitive-Emotional integration
        final_state, integration_metrics = self.cognitive_emotional(perspectival_state, emotional_state)
        all_metrics['integration'] = integration_metrics
        
        # 7. Determine processing mode
        novelty = 1.0 - all_metrics['integration']['meaning_coherence']
        coherence = all_metrics['integration']['meaning_coherence']
        mode = self.mode_manager.select_mode(novelty, coherence)
        all_metrics['processing_mode'] = mode.value
        
        # 8. Compute overall framework metrics
        all_metrics['framework'] = self._compute_framework_metrics(all_metrics)
        
        return final_state, all_metrics
    
    def _compute_framework_metrics(self, all_metrics: Dict) -> Dict[str, float]:
        """Compute overall framework performance metrics"""
        # Wisdom measure: integration of all 4E components
        wisdom = (
            all_metrics['embodied']['sensorimotor_coupling'] * 0.25 +
            all_metrics['embedded']['environmental_coupling'] * 0.25 +
            all_metrics['enacted']['exploration_intensity'] * 0.25 +
            all_metrics['extended']['cognitive_extension'] * 0.25
        )
        
        # Meaning connectivity (anti-meaning-crisis measure)
        meaning = (
            all_metrics['perspectival']['participation_depth'] * 0.4 +
            all_metrics['integration']['meaning_coherence'] * 0.6
        )
        
        # Overall coherence
        coherence = all_metrics['integration']['integration_quality']
        
        return {
            'wisdom_measure': wisdom,
            'meaning_connectivity': meaning,
            'overall_coherence': coherence,
            'four_e_integration': (wisdom + meaning + coherence) / 3.0
        }
