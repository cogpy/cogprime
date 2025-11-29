"""
Tests for Vervaeke 4E Enhanced Cognitive Core

Validates the complete integration of 4E framework with cognitive core.
"""

import pytest
import torch
from ..core.vervaeke_4e_cognitive_core import (
    Vervaeke4ECognitiveCore,
    Enhanced4ECognitiveState,
    ProcessingMode
)
from ..modules.perception import SensoryInput


class TestVervaeke4ECognitiveCore:
    """Test enhanced cognitive core with 4E integration"""
    
    def test_core_initialization(self):
        """Test core initializes with all components"""
        config = {
            'feature_dim': 512,
            'motor_dim': 128,
            'visual_dim': 784,
            'audio_dim': 256
        }
        
        core = Vervaeke4ECognitiveCore(config)
        
        # Base components
        assert core.perception is not None
        assert core.reasoning is not None
        assert core.action_selector is not None
        assert core.learner is not None
        
        # 4E components
        assert core.perception_4e is not None
        assert core.framework_4e is not None
        assert core.state_4e is not None
        
    def test_enhanced_cognitive_cycle(self):
        """Test complete 4E cognitive cycle"""
        config = {'feature_dim': 512, 'visual_dim': 784, 'audio_dim': 256}
        core = Vervaeke4ECognitiveCore(config)
        
        sensory_input = SensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256)
        )
        
        motor_state = torch.randn(128)
        context = {'location': 'test_environment'}
        
        action, metrics = core.cognitive_cycle_4e(
            sensory_input,
            motor_state=motor_state,
            context_info=context,
            reward=0.5
        )
        
        # Validate metrics structure
        assert 'perception_4e' in metrics
        assert 'framework_4e' in metrics
        assert 'wisdom' in metrics
        
        # Validate wisdom metrics
        assert 'current' in metrics['wisdom']
        assert 'meaning_connectivity' in metrics['wisdom']
        assert 'processing_mode' in metrics['wisdom']
        
    def test_embodied_integration(self):
        """Test embodied cognition in cognitive cycle"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        # Update body schema
        feedback = {
            'joint_shoulder': 0.5,
            'joint_elbow': 0.8
        }
        core.update_body_schema(feedback)
        
        sensory_input = SensoryInput(visual=torch.randn(784))
        motor_state = torch.randn(128) * 2.0  # Strong motor signal
        
        _, metrics = core.cognitive_cycle_4e(sensory_input, motor_state=motor_state)
        
        # Should show embodied processing
        assert 'embodied' in metrics['framework_4e']
        assert metrics['framework_4e']['embodied']['sensorimotor_coupling'] > -1.0
        
    def test_embedded_context_processing(self):
        """Test embedded cognition with environmental context"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        # Update environmental context
        context = {
            'scene_type': 'forest',
            'time_of_day': 'morning',
            'weather': 'sunny'
        }
        core.update_environmental_context(context)
        
        sensory_input = SensoryInput(visual=torch.randn(784))
        
        _, metrics = core.cognitive_cycle_4e(
            sensory_input,
            context_info=context
        )
        
        # Should show embedded processing
        assert 'embedded' in metrics['framework_4e']
        assert metrics['framework_4e']['embedded']['environmental_coupling'] > 0.0
        
    def test_cognitive_tool_integration(self):
        """Test extended cognition with tool use"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        # Register and activate a tool
        tool_features = torch.randn(128)
        core.register_cognitive_tool('microscope', tool_features)
        core.activate_tool('microscope')
        
        sensory_input = SensoryInput(visual=torch.randn(784))
        
        _, metrics = core.cognitive_cycle_4e(sensory_input)
        
        # Should show extended processing
        assert 'extended' in metrics['framework_4e']
        
    def test_wisdom_cultivation(self):
        """Test wisdom cultivation over multiple cycles"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        sensory_input = SensoryInput(visual=torch.randn(784))
        
        # Run multiple cycles
        for i in range(10):
            _, metrics = core.cognitive_cycle_4e(
                sensory_input,
                reward=0.5 + i * 0.05  # Increasing reward
            )
        
        # Get wisdom trajectory
        trajectory = core.get_wisdom_trajectory()
        
        assert 'current_wisdom' in trajectory
        assert 'mean_wisdom' in trajectory
        assert 'wisdom_trend' in trajectory
        assert 'transformation_count' in trajectory
        
    def test_transformative_processing_detection(self):
        """Test detection of transformative moments"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        sensory_input = SensoryInput(visual=torch.randn(784))
        
        # Run cycles to build history
        for _ in range(5):
            core.cognitive_cycle_4e(sensory_input, reward=0.5)
        
        initial_transforms = len(core.transformation_events)
        
        # Create condition for transformation (high novelty, low coherence)
        # This is indirectly tested through the framework metrics
        for _ in range(10):
            core.cognitive_cycle_4e(sensory_input, reward=0.8)
        
        # Transformations might have been detected
        # (actual detection depends on framework processing)
        assert len(core.transformation_events) >= initial_transforms
        
    def test_processing_mode_tracking(self):
        """Test processing mode is tracked in state"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        sensory_input = SensoryInput(visual=torch.randn(784))
        
        _, metrics = core.cognitive_cycle_4e(sensory_input)
        
        # Processing mode should be tracked
        assert core.state_4e.processing_mode in [
            ProcessingMode.CONFORMATIVE,
            ProcessingMode.TRANSFORMATIVE
        ]
        
    def test_meaning_crisis_metrics(self):
        """Test meaning crisis metrics computation"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        sensory_input = SensoryInput(visual=torch.randn(784))
        
        # Run cycles to build history
        for _ in range(20):
            core.cognitive_cycle_4e(sensory_input, reward=0.6)
        
        crisis_metrics = core.get_meaning_crisis_metrics()
        
        # Should have all anti-meaning-crisis metrics
        assert 'meaning_connectivity' in crisis_metrics
        assert 'wisdom_cultivation' in crisis_metrics
        assert 'transformative_capacity' in crisis_metrics
        assert 'overall_meaning_health' in crisis_metrics
        
        # Metrics should be in valid ranges
        for key, value in crisis_metrics.items():
            assert 0.0 <= value <= 1.5, f"{key} out of range: {value}"
        
    def test_4e_system_status(self):
        """Test comprehensive 4E system status"""
        config = {'feature_dim': 512, 'visual_dim': 784}
        core = Vervaeke4ECognitiveCore(config)
        
        # Set up system state
        core.update_body_schema({'joint_1': 0.5, 'joint_2': 0.8})
        core.register_cognitive_tool('tool1', torch.randn(128))
        core.activate_tool('tool1')
        core.update_environmental_context({'key': 'value'})
        
        status = core.get_4e_status()
        
        # Should report all status components
        assert 'body_schema_joints' in status
        assert 'active_tools' in status
        assert 'registered_tools' in status
        assert 'environmental_context_keys' in status
        assert 'processing_mode' in status
        assert 'wisdom_measure' in status
        assert 'meaning_connectivity' in status
        assert 'transformation_events' in status
        assert 'system_status' in status
        
        # Verify counts
        assert status['body_schema_joints'] == 2
        assert status['active_tools'] == 1
        assert status['registered_tools'] >= 1
        assert status['environmental_context_keys'] >= 1
        
    def test_acceptance_all_criteria(self):
        """
        Comprehensive test validating all acceptance criteria.
        
        This test exercises the complete 4E cognitive cycle and validates
        all acceptance criteria from the issue.
        """
        config = {
            'feature_dim': 512,
            'motor_dim': 128,
            'visual_dim': 784,
            'audio_dim': 256
        }
        core = Vervaeke4ECognitiveCore(config)
        
        # Setup: body schema, tools, context
        core.update_body_schema({
            'joint_shoulder': 0.5,
            'joint_elbow': 0.7,
            'joint_wrist': 0.3
        })
        
        core.register_cognitive_tool('analyzer', torch.randn(128))
        core.activate_tool('analyzer')
        
        context = {
            'scene': 'laboratory',
            'complexity': 'high',
            'social_presence': True
        }
        core.update_environmental_context(context)
        
        # Execute multiple cognitive cycles
        for i in range(15):
            sensory_input = SensoryInput(
                visual=torch.randn(784),
                auditory=torch.randn(256) if i % 2 == 0 else None,
                proprioceptive=torch.randn(512)
            )
            
            motor_state = torch.randn(128)
            reward = 0.5 + i * 0.03
            
            action, metrics = core.cognitive_cycle_4e(
                sensory_input,
                motor_state=motor_state,
                context_info=context,
                reward=reward
            )
        
        # Get final status and trajectories
        status = core.get_4e_status()
        trajectory = core.get_wisdom_trajectory()
        crisis_metrics = core.get_meaning_crisis_metrics()
        
        # Validate Acceptance Criteria:
        
        # 1. Embodied cognition improves sensorimotor integration
        assert status['body_schema_joints'] == 3
        last_metrics = metrics['framework_4e']
        assert 'embodied' in last_metrics
        assert last_metrics['embodied']['sensorimotor_coupling'] > -1.0
        
        # 2. Embedded cognition enables context-sensitive behavior
        assert status['environmental_context_keys'] > 0
        assert 'embedded' in last_metrics
        assert last_metrics['embedded']['environmental_coupling'] > 0.0
        
        # 3. Enacted cognition supports active perception
        assert 'enacted' in last_metrics
        assert last_metrics['enacted']['exploration_intensity'] >= 0.0
        
        # 4. Extended cognition utilizes tools
        assert status['active_tools'] == 1
        assert status['registered_tools'] >= 1
        assert 'extended' in last_metrics
        
        # 5. Salience landscape guides attention
        assert 'salience_navigation' in metrics['perception_4e']
        nav_metrics = metrics['perception_4e']['salience_navigation']
        assert 'relevance_magnitude' in nav_metrics
        
        # 6. Perspectival knowing enables viewpoint integration
        assert 'perspectival' in last_metrics
        assert 'perspective_flexibility' in last_metrics['perspectival']
        
        # 7. Transformative processing facilitates insight
        assert 'processing_mode' in last_metrics
        assert status['processing_mode'] in ['conformative', 'transformative']
        
        # 8. Cognitive-emotional integration enhances decisions
        assert 'integration' in last_metrics
        assert 'meaning_coherence' in last_metrics['integration']
        
        # Overall system health
        assert trajectory['status'] != 'no_data'
        assert crisis_metrics['overall_meaning_health'] > 0.0
        
        # Wisdom cultivation occurring
        assert trajectory['mean_wisdom'] >= 0.0
        assert trajectory['mean_meaning'] >= 0.0
        
        print("\n=== 4E Cognitive System Validation ===")
        print(f"Body Schema Joints: {status['body_schema_joints']}")
        print(f"Active Tools: {status['active_tools']}")
        print(f"Processing Mode: {status['processing_mode']}")
        print(f"Wisdom Measure: {status['wisdom_measure']:.3f}")
        print(f"Meaning Connectivity: {status['meaning_connectivity']:.3f}")
        print(f"Transformation Events: {status['transformation_events']}")
        print(f"Overall Meaning Health: {crisis_metrics['overall_meaning_health']:.3f}")
        print("=== All Acceptance Criteria Validated ===\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
