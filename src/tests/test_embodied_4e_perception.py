"""
Tests for Enhanced 4E Perception Integration

Validates the integration of Vervaeke's 4E framework with the perception module.
"""

import pytest
import torch
from ..integration.embodied_4e_perception import (
    Embodied4EPerceptionModule,
    EnhancedSensoryInput,
    SalienceLandscapeNavigator,
    Embodied4EPerceptionSystem
)


class TestEmbodied4EPerception:
    """Test enhanced 4E perception module"""
    
    def test_module_initialization(self):
        """Test module initializes all components"""
        config = {
            'feature_dim': 512,
            'motor_dim': 128,
            'visual_dim': 784,
            'audio_dim': 256
        }
        
        module = Embodied4EPerceptionModule(config)
        
        assert module.perception is not None
        assert module.embodied_processor is not None
        assert module.embedded_processor is not None
        assert module.enacted_processor is not None
        assert module.extended_processor is not None
        
    def test_basic_perception_processing(self):
        """Test basic perception with 4E integration"""
        config = {'feature_dim': 512, 'visual_dim': 784, 'audio_dim': 256}
        module = Embodied4EPerceptionModule(config)
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256),
            motor_state=torch.randn(128)
        )
        
        perception, metrics = module(enhanced_input)
        
        # Validate output shape
        assert perception.shape == (512,)
        
        # Validate metrics structure
        assert 'perception' in metrics
        assert 'embodied' in metrics
        assert 'embedded' in metrics
        assert 'enacted' in metrics
        assert 'extended' in metrics
        assert 'four_e_integration' in metrics
        
    def test_embodied_sensorimotor_integration(self):
        """Test embodied cognition improves with motor state"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        module = Embodied4EPerceptionModule(config)
        
        # Strong motor signal
        input_active = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128) * 2.0
        )
        
        # Weak motor signal
        input_passive = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128) * 0.1
        )
        
        _, metrics_active = module(input_active)
        _, metrics_passive = module(input_passive)
        
        # Active motor should show stronger sensorimotor grounding
        assert metrics_active['four_e_integration']['sensorimotor_grounding'] > \
               metrics_passive['four_e_integration']['sensorimotor_grounding']
               
    def test_embedded_context_awareness(self):
        """Test embedded cognition responds to context"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        module = Embodied4EPerceptionModule(config)
        
        context_rich = {'scene_type': 'indoor', 'lighting': 'bright', 'obstacles': 'few'}
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128),
            context_info=context_rich
        )
        
        _, metrics = module(enhanced_input)
        
        # Should show contextual adaptation
        assert 'contextual_adaptation' in metrics['four_e_integration']
        assert metrics['embedded']['environmental_coupling'] > 0.0
        
    def test_enacted_active_perception(self):
        """Test enacted cognition with action"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        module = Embodied4EPerceptionModule(config)
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128)
        )
        
        action = torch.randn(64)
        
        _, metrics = module(enhanced_input, action=action)
        
        # Should show exploration engagement
        assert 'exploration_engagement' in metrics['four_e_integration']
        assert metrics['enacted']['exploration_intensity'] > 0.0
        
    def test_extended_tool_integration(self):
        """Test extended cognition with tool use"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        module = Embodied4EPerceptionModule(config)
        
        # Register a tool
        tool_features = torch.randn(128)
        module.register_tool('telescope', tool_features)
        
        # Use the tool
        tool_state = module.use_tool('telescope')
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128)
        )
        
        _, metrics = module(enhanced_input, tool_state=tool_state)
        
        # Should show extension utilization
        assert 'extension_utilization' in metrics['four_e_integration']
        assert metrics['extended']['tool_usage'] > 0.0
        
    def test_body_schema_update(self):
        """Test body schema updates from feedback"""
        config = {'feature_dim': 512}
        module = Embodied4EPerceptionModule(config)
        
        feedback = {
            'joint_shoulder': 0.5,
            'joint_elbow': 0.8
        }
        
        module.update_body_schema(feedback)
        
        # Check body schema was updated
        assert 'joint_shoulder' in module.embodied_processor.body_schema.joint_states
        assert module.embodied_processor.body_schema.joint_states['joint_shoulder'] == 0.5
        
    def test_overall_4e_integration_quality(self):
        """Test overall 4E integration produces quality metrics"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        module = Embodied4EPerceptionModule(config)
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128),
            context_info={'type': 'test'}
        )
        
        action = torch.randn(64)
        tool_state = torch.randn(128)
        
        _, metrics = module(enhanced_input, action=action, tool_state=tool_state)
        
        integration = metrics['four_e_integration']
        
        # All integration metrics should be present
        assert 'sensorimotor_grounding' in integration
        assert 'contextual_adaptation' in integration
        assert 'exploration_engagement' in integration
        assert 'extension_utilization' in integration
        assert 'overall_4e_integration' in integration
        
        # Overall integration should be reasonable
        assert 0.0 <= integration['overall_4e_integration'] <= 2.0


class TestSalienceLandscapeNavigator:
    """Test salience landscape navigation"""
    
    def test_navigator_initialization(self):
        """Test navigator initializes correctly"""
        navigator = SalienceLandscapeNavigator(feature_dim=512, salience_dim=256)
        
        assert navigator.feature_dim == 512
        assert navigator.salience_dim == 256
        
    def test_salience_field_generation(self):
        """Test salience field generation"""
        navigator = SalienceLandscapeNavigator()
        
        perception = torch.randn(512)
        metrics = {'overall_4e_integration': 0.8}
        
        attention_target, nav_metrics = navigator(perception, metrics)
        
        # Validate output
        assert attention_target.shape == (512,)
        
        # Validate metrics
        assert 'peak_strength' in nav_metrics
        assert 'landscape_coherence' in nav_metrics
        assert 'salience_intensity' in nav_metrics
        assert 'relevance_magnitude' in nav_metrics
        
    def test_4e_modulation_of_salience(self):
        """Test 4E metrics modulate salience"""
        navigator = SalienceLandscapeNavigator()
        
        perception = torch.randn(512)
        
        # High integration quality
        metrics_high = {'overall_4e_integration': 0.9}
        attention_high, nav_metrics_high = navigator(perception, metrics_high)
        
        # Low integration quality
        metrics_low = {'overall_4e_integration': 0.1}
        attention_low, nav_metrics_low = navigator(perception, metrics_low)
        
        # Higher integration should produce stronger salience
        assert nav_metrics_high['salience_intensity'] > nav_metrics_low['salience_intensity']
        
    def test_landscape_coherence_tracking(self):
        """Test landscape coherence over time"""
        navigator = SalienceLandscapeNavigator()
        
        # Process multiple cycles
        for i in range(5):
            perception = torch.randn(512)
            metrics = {'overall_4e_integration': 0.7}
            _, nav_metrics = navigator(perception, metrics)
        
        # Should compute coherence
        assert 'landscape_coherence' in nav_metrics
        assert 0.0 <= nav_metrics['landscape_coherence'] <= 1.0
        
    def test_attention_peak_detection(self):
        """Test attention peak detection"""
        navigator = SalienceLandscapeNavigator()
        
        perception = torch.randn(512)
        metrics = {'overall_4e_integration': 0.8}
        
        _, nav_metrics = navigator(perception, metrics)
        
        # Should detect peaks
        assert 'peak_strength' in nav_metrics
        assert 0.0 <= nav_metrics['peak_strength'] <= 1.0


class TestEmbodied4EPerceptionSystem:
    """Test complete 4E perception system"""
    
    def test_system_initialization(self):
        """Test system initializes all components"""
        config = {
            'feature_dim': 512,
            'visual_dim': 784,
            'audio_dim': 256
        }
        
        system = Embodied4EPerceptionSystem(config)
        
        assert system.perception_4e is not None
        assert system.salience_navigator is not None
        
    def test_complete_perception_cycle(self):
        """Test complete perception and navigation cycle"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        system = Embodied4EPerceptionSystem(config)
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128)
        )
        
        perception, attention, metrics = system.process_and_navigate(enhanced_input)
        
        # Validate outputs
        assert perception.shape == (512,)
        assert attention.shape == (512,)
        
        # Validate comprehensive metrics
        assert 'perception' in metrics
        assert 'embodied' in metrics
        assert 'embedded' in metrics
        assert 'enacted' in metrics
        assert 'extended' in metrics
        assert 'four_e_integration' in metrics
        assert 'salience_navigation' in metrics
        
    def test_acceptance_spatial_reasoning(self):
        """Validate: Embodied cognition improves spatial reasoning"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        system = Embodied4EPerceptionSystem(config)
        
        # Strong proprioceptive signal for spatial reasoning
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            proprioceptive=torch.randn(512),
            motor_state=torch.randn(128) * 2.0
        )
        
        _, _, metrics = system.process_and_navigate(enhanced_input)
        
        # Should show strong spatial awareness
        assert metrics['embodied']['spatial_awareness'] > 0.5
        
    def test_acceptance_context_sensitive_behavior(self):
        """Validate: Embedded cognition enables context-sensitive behavior"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        system = Embodied4EPerceptionSystem(config)
        
        # Process with rich context
        context1 = {'scene': 'forest', 'time': 'day'}
        input1 = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128),
            context_info=context1
        )
        
        context2 = {'scene': 'urban', 'time': 'night'}
        input2 = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128),
            context_info=context2
        )
        
        _, _, metrics1 = system.process_and_navigate(input1)
        _, _, metrics2 = system.process_and_navigate(input2)
        
        # Different contexts should produce different metrics
        assert metrics1['embedded']['environmental_coupling'] != \
               metrics2['embedded']['environmental_coupling']
               
    def test_acceptance_active_exploration(self):
        """Validate: Enacted cognition supports active exploration"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        system = Embodied4EPerceptionSystem(config)
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128)
        )
        
        # Provide exploratory action
        exploratory_action = torch.randn(64) * 2.0
        
        _, _, metrics = system.process_and_navigate(
            enhanced_input,
            action=exploratory_action
        )
        
        # Should show strong exploration
        assert metrics['enacted']['exploration_intensity'] > 0.5
        
    def test_acceptance_environmental_affordances(self):
        """Validate: Extended cognition utilizes environmental affordances"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        system = Embodied4EPerceptionSystem(config)
        
        # Register multiple tools
        system.register_tool('magnifier', torch.randn(128))
        system.register_tool('compass', torch.randn(128))
        
        # Use a tool
        tool_state = system.use_tool('magnifier')
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128)
        )
        
        _, _, metrics = system.process_and_navigate(
            enhanced_input,
            tool_state=tool_state
        )
        
        # Should show tool usage
        assert metrics['extended']['tool_usage'] > 0.0
        
    def test_acceptance_salience_guidance(self):
        """Validate: Salience landscape guides attention and action"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        system = Embodied4EPerceptionSystem(config)
        
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128)
        )
        
        _, attention_target, metrics = system.process_and_navigate(enhanced_input)
        
        # Should provide attention guidance
        assert attention_target is not None
        assert metrics['salience_navigation']['relevance_magnitude'] > 0.0
        
        # Attention should be influenced by salience peaks
        assert 'peak_strength' in metrics['salience_navigation']
        
    def test_tool_management(self):
        """Test tool registration and usage"""
        config = {'feature_dim': 512}
        system = Embodied4EPerceptionSystem(config)
        
        # Register tools
        system.register_tool('tool1', torch.randn(128))
        system.register_tool('tool2', torch.randn(128))
        
        status = system.get_system_status()
        
        assert status['registered_tools'] == 2
        
    def test_body_schema_management(self):
        """Test body schema updates"""
        config = {'feature_dim': 512}
        system = Embodied4EPerceptionSystem(config)
        
        feedback = {'joint_1': 0.5, 'joint_2': 0.8, 'joint_3': 0.3}
        system.update_body_schema(feedback)
        
        status = system.get_system_status()
        
        assert status['body_schema_joints'] == 3
        
    def test_system_status_reporting(self):
        """Test comprehensive system status"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        system = Embodied4EPerceptionSystem(config)
        
        # Process some inputs
        enhanced_input = EnhancedSensoryInput(
            visual=torch.randn(784),
            motor_state=torch.randn(128)
        )
        
        for _ in range(3):
            system.process_and_navigate(enhanced_input)
        
        status = system.get_system_status()
        
        # Should track all components
        assert 'body_schema_joints' in status
        assert 'registered_tools' in status
        assert 'action_history_size' in status
        assert 'salience_history_size' in status
        assert status['salience_history_size'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
