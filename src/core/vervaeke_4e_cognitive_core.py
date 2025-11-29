"""
Enhanced Cognitive Core with Vervaeke 4E Framework Integration

Extends the CogPrimeCore with full 4E cognition capabilities including:
- Embodied sensorimotor processing
- Embedded contextual awareness
- Enacted active perception
- Extended cognitive distribution
- Perspectival/participatory knowing
- Transformative/conformative processing
- Cognitive-emotional integration
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..modules.perception import SensoryInput
from ..modules.reasoning import Thought
from ..modules.action import Action
from ..core.cognitive_core import CogPrimeCore, CognitiveState
from ..integration.embodied_4e_perception import (
    Embodied4EPerceptionSystem,
    EnhancedSensoryInput
)
from ..cognitive_science.vervaeke_4e_framework import (
    Vervaeke4EFramework,
    ProcessingMode,
    PerspectivalParticipatoryProcessor,
    CognitiveEmotionalIntegrator
)


@dataclass
class Enhanced4ECognitiveState(CognitiveState):
    """Extended cognitive state with 4E cognition components"""
    motor_state: torch.Tensor = None
    emotional_state: torch.Tensor = None
    body_schema_state: Dict[str, float] = field(default_factory=dict)
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)
    processing_mode: ProcessingMode = ProcessingMode.CONFORMATIVE
    four_e_metrics: Dict[str, Any] = field(default_factory=dict)
    wisdom_measure: float = 0.0
    meaning_connectivity: float = 0.0


class Vervaeke4ECognitiveCore(CogPrimeCore):
    """
    Enhanced cognitive core integrating Vervaeke's 4E framework.
    
    Extends CogPrimeCore with embodied, embedded, enacted, and extended
    cognition for addressing the meaning crisis through wisdom cultivation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Initialize base cognitive core
        super().__init__(config)
        
        # Initialize 4E perception system
        self.perception_4e = Embodied4EPerceptionSystem(config)
        
        # Initialize complete 4E framework
        self.framework_4e = Vervaeke4EFramework(config)
        
        # Enhanced cognitive state
        self.state_4e = Enhanced4ECognitiveState(
            attention_focus=torch.zeros(512),
            working_memory={},
            emotional_valence=0.0,
            goal_stack=[],
            sensory_buffer={},
            motor_state=torch.zeros(config.get('motor_dim', 128)),
            emotional_state=torch.zeros(config.get('emotion_dim', 64)),
            body_schema_state={},
            environmental_context={},
            active_tools=[],
            processing_mode=ProcessingMode.CONFORMATIVE,
            four_e_metrics={},
            wisdom_measure=0.0,
            meaning_connectivity=0.0
        )
        
        # Wisdom cultivation tracking
        self.wisdom_history = []
        self.meaning_history = []
        self.transformation_events = []
        
    def cognitive_cycle_4e(self,
                          sensory_input: SensoryInput,
                          motor_state: Optional[torch.Tensor] = None,
                          context_info: Optional[Dict] = None,
                          reward: float = 0.0) -> Tuple[Optional[Action], Dict[str, Any]]:
        """
        Execute enhanced cognitive cycle with 4E processing.
        
        Args:
            sensory_input: Standard sensory input
            motor_state: Optional motor/body state
            context_info: Optional environmental context
            reward: Reward signal
            
        Returns:
            action: Selected action
            cycle_metrics: Comprehensive 4E cycle metrics
        """
        cycle_metrics = {}
        
        # 1. Enhanced 4E Perception
        enhanced_input = EnhancedSensoryInput(
            visual=sensory_input.visual,
            auditory=sensory_input.auditory,
            proprioceptive=sensory_input.proprioceptive,
            motor_state=motor_state if motor_state is not None else self.state_4e.motor_state,
            emotional_state=self.state_4e.emotional_state,
            text=sensory_input.text,
            context_info=context_info or self.state_4e.environmental_context
        )
        
        perception_state, attention_target, perception_metrics = \
            self.perception_4e.process_and_navigate(enhanced_input)
        
        cycle_metrics['perception_4e'] = perception_metrics
        
        # 2. Complete 4E Framework Processing
        framework_state, framework_metrics = self.framework_4e.process_cycle(
            sensory_input=perception_state,
            motor_state=self.state_4e.motor_state,
            emotional_state=self.state_4e.emotional_state,
            context_info=context_info,
            action=None,  # Will be determined by action selection
            tool_state=self._get_active_tool_state()
        )
        
        cycle_metrics['framework_4e'] = framework_metrics
        
        # 3. Update cognitive state
        self._update_4e_state(
            framework_state,
            attention_target,
            perception_metrics,
            framework_metrics
        )
        
        # 4. Reasoning with 4E-informed state
        self._reason_4e(framework_state, framework_metrics)
        
        # 5. Action selection guided by salience and 4E metrics
        action = self._act_4e(attention_target, framework_metrics)
        
        # 6. Learning from 4E experience
        if action:
            self._learn_4e(reward, framework_metrics)
        
        # 7. Wisdom cultivation and meaning tracking
        self._cultivate_wisdom(framework_metrics)
        
        # 8. Detect and process transformative moments
        self._detect_transformation(framework_metrics)
        
        # Compile comprehensive cycle metrics
        cycle_metrics['wisdom'] = {
            'current': self.state_4e.wisdom_measure,
            'meaning_connectivity': self.state_4e.meaning_connectivity,
            'processing_mode': self.state_4e.processing_mode.value
        }
        
        return action, cycle_metrics
    
    def _update_4e_state(self,
                         framework_state: torch.Tensor,
                         attention_target: torch.Tensor,
                         perception_metrics: Dict,
                         framework_metrics: Dict):
        """Update 4E cognitive state from processing cycle"""
        
        # Update attention
        self.state_4e.attention_focus = attention_target
        
        # Store 4E metrics
        self.state_4e.four_e_metrics = {
            'perception': perception_metrics,
            'framework': framework_metrics
        }
        
        # Update processing mode
        mode_str = framework_metrics.get('processing_mode', 'conformative')
        self.state_4e.processing_mode = ProcessingMode(mode_str)
        
        # Update wisdom and meaning measures
        if 'framework' in framework_metrics:
            framework_summary = framework_metrics['framework']
            self.state_4e.wisdom_measure = framework_summary.get('wisdom_measure', 0.0)
            self.state_4e.meaning_connectivity = framework_summary.get('meaning_connectivity', 0.0)
        
        # Update sensory buffer
        self.state_4e.sensory_buffer = {
            'framework_state': framework_state,
            'attention_target': attention_target,
            '4e_metrics': framework_metrics
        }
    
    def _reason_4e(self, framework_state: torch.Tensor, framework_metrics: Dict):
        """Enhanced reasoning with 4E context"""
        
        # Use base reasoning but inform with 4E state
        self.state_4e.working_memory['4e_context'] = {
            'embodied': framework_metrics.get('embodied', {}),
            'embedded': framework_metrics.get('embedded', {}),
            'enacted': framework_metrics.get('enacted', {}),
            'extended': framework_metrics.get('extended', {})
        }
        
        # Process thought with 4E-informed working memory
        thought, updated_memory = self.reasoning.process_thought(
            framework_state,
            self.state_4e.working_memory
        )
        
        self.state_4e.current_thought = thought
        self.state_4e.working_memory = updated_memory
    
    def _act_4e(self, attention_target: torch.Tensor, framework_metrics: Dict) -> Optional[Action]:
        """Action selection guided by 4E salience and metrics"""
        
        # Modulate action selection with 4E integration quality
        integration_quality = framework_metrics.get('framework', {}).get(
            'four_e_integration', 0.5
        )
        
        # Use attention target to guide action selection
        features = attention_target
        
        # Get action from action selector
        action = self.action_selector.select_action(
            features,
            self.state_4e.working_memory,
            available_resources=100.0 * integration_quality
        )
        
        self.state_4e.last_action = action
        return action
    
    def _learn_4e(self, reward: float, framework_metrics: Dict):
        """Learning enhanced with 4E understanding"""
        
        # Base learning
        if self.state_4e.last_action:
            # Modulate reward with wisdom and meaning measures
            wisdom_bonus = self.state_4e.wisdom_measure * 0.1
            meaning_bonus = self.state_4e.meaning_connectivity * 0.1
            
            enhanced_reward = reward + wisdom_bonus + meaning_bonus
            
            # Update learner with enhanced reward
            self.state_4e.last_reward = enhanced_reward
            self.state_4e.total_reward += enhanced_reward
            
            # Store in working memory
            self.state_4e.working_memory['last_reward'] = enhanced_reward
            self.state_4e.working_memory['wisdom_bonus'] = wisdom_bonus
            self.state_4e.working_memory['meaning_bonus'] = meaning_bonus
    
    def _cultivate_wisdom(self, framework_metrics: Dict):
        """Track and cultivate wisdom over time"""
        
        if 'framework' in framework_metrics:
            wisdom = framework_metrics['framework'].get('wisdom_measure', 0.0)
            meaning = framework_metrics['framework'].get('meaning_connectivity', 0.0)
            
            self.wisdom_history.append(wisdom)
            self.meaning_history.append(meaning)
            
            # Keep only recent history
            if len(self.wisdom_history) > 100:
                self.wisdom_history.pop(0)
                self.meaning_history.pop(0)
    
    def _detect_transformation(self, framework_metrics: Dict):
        """Detect and record transformative moments"""
        
        # Transformative mode indicates potential insight or paradigm shift
        if framework_metrics.get('processing_mode') == 'transformative':
            
            # Check for significant change in wisdom or meaning
            if len(self.wisdom_history) > 1:
                wisdom_change = abs(
                    self.wisdom_history[-1] - np.mean(self.wisdom_history[:-1])
                )
                
                if wisdom_change > 0.1:  # Significant threshold
                    transformation = {
                        'cycle': len(self.wisdom_history),
                        'wisdom_change': wisdom_change,
                        'meaning_level': self.state_4e.meaning_connectivity,
                        'processing_mode': 'transformative'
                    }
                    self.transformation_events.append(transformation)
    
    def _get_active_tool_state(self) -> Optional[torch.Tensor]:
        """Get state of currently active tools"""
        if not self.state_4e.active_tools:
            return None
        
        # Use the first active tool (could be extended to combine multiple)
        tool_name = self.state_4e.active_tools[0]
        return self.perception_4e.use_tool(tool_name)
    
    def register_cognitive_tool(self, tool_name: str, tool_features: torch.Tensor):
        """Register a new cognitive tool"""
        self.perception_4e.register_tool(tool_name, tool_features)
        self.framework_4e.extended.register_tool(tool_name, tool_features)
    
    def activate_tool(self, tool_name: str):
        """Activate a cognitive tool"""
        if tool_name not in self.state_4e.active_tools:
            self.state_4e.active_tools.append(tool_name)
    
    def deactivate_tool(self, tool_name: str):
        """Deactivate a cognitive tool"""
        if tool_name in self.state_4e.active_tools:
            self.state_4e.active_tools.remove(tool_name)
    
    def update_body_schema(self, feedback: Dict[str, float]):
        """Update body schema from sensory feedback"""
        self.perception_4e.update_body_schema(feedback)
        self.framework_4e.embodied.update_body_schema(feedback)
        self.state_4e.body_schema_state.update(feedback)
    
    def update_environmental_context(self, context: Dict[str, Any]):
        """Update environmental context"""
        self.state_4e.environmental_context.update(context)
    
    def get_wisdom_trajectory(self) -> Dict[str, Any]:
        """Get wisdom cultivation trajectory"""
        if not self.wisdom_history:
            return {'status': 'no_data'}
        
        return {
            'current_wisdom': self.wisdom_history[-1],
            'mean_wisdom': np.mean(self.wisdom_history),
            'wisdom_trend': np.polyfit(
                range(len(self.wisdom_history)),
                self.wisdom_history,
                1
            )[0] if len(self.wisdom_history) > 2 else 0.0,
            'current_meaning': self.meaning_history[-1],
            'mean_meaning': np.mean(self.meaning_history),
            'transformation_count': len(self.transformation_events)
        }
    
    def get_4e_status(self) -> Dict[str, Any]:
        """Get comprehensive 4E cognitive system status"""
        return {
            'body_schema_joints': len(self.state_4e.body_schema_state),
            'active_tools': len(self.state_4e.active_tools),
            'registered_tools': len(self.framework_4e.extended.tool_mastery),
            'environmental_context_keys': len(self.state_4e.environmental_context),
            'processing_mode': self.state_4e.processing_mode.value,
            'wisdom_measure': self.state_4e.wisdom_measure,
            'meaning_connectivity': self.state_4e.meaning_connectivity,
            'transformation_events': len(self.transformation_events),
            'system_status': self.perception_4e.get_system_status()
        }
    
    def get_meaning_crisis_metrics(self) -> Dict[str, float]:
        """
        Get metrics related to the meaning crisis.
        
        Higher values indicate better addressing of meaning crisis.
        """
        if not self.meaning_history:
            return {
                'meaning_connectivity': 0.0,
                'wisdom_cultivation': 0.0,
                'transformative_capacity': 0.0,
                'overall_meaning_health': 0.0
            }
        
        # Meaning connectivity (connection to meaning)
        meaning_conn = np.mean(self.meaning_history[-10:])
        
        # Wisdom cultivation (growth in wisdom)
        wisdom_cult = np.mean(self.wisdom_history[-10:])
        
        # Transformative capacity (ability to transform when needed)
        total_cycles = len(self.wisdom_history)
        transform_ratio = len(self.transformation_events) / max(total_cycles, 1)
        transformative_cap = min(1.0, transform_ratio * 10)  # Scale to 0-1
        
        # Overall meaning health
        overall = (meaning_conn + wisdom_cult + transformative_cap) / 3.0
        
        return {
            'meaning_connectivity': meaning_conn,
            'wisdom_cultivation': wisdom_cult,
            'transformative_capacity': transformative_cap,
            'overall_meaning_health': overall
        }
