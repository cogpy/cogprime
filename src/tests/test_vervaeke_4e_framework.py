"""
Tests for Vervaeke 4E Cognition Framework

Validates all components of the 4E framework implementation including:
- Embodied cognition processing
- Embedded cognition context awareness
- Enacted cognition active perception
- Extended cognition tool use
- Perspectival/Participatory knowing
- Processing mode management
- Cognitive-emotional integration
- Complete framework integration
"""

import pytest
import torch
import numpy as np
from ..cognitive_science.vervaeke_4e_framework import (
    EmbodiedCognitionProcessor,
    EmbeddedCognitionProcessor,
    EnactedCognitionProcessor,
    ExtendedCognitionProcessor,
    PerspectivalParticipatoryProcessor,
    ProcessingModeManager,
    ProcessingMode,
    CognitiveEmotionalIntegrator,
    Vervaeke4EFramework,
    BodySchema,
    EnvironmentalContext,
)


class TestEmbodiedCognition:
    """Test embodied cognition processor"""
    
    def test_embodied_initialization(self):
        """Test processor initializes correctly"""
        processor = EmbodiedCognitionProcessor(sensory_dim=512, motor_dim=128)
        assert processor.sensory_dim == 512
        assert processor.motor_dim == 128
        assert processor.body_schema is not None
        
    def test_embodied_processing(self):
        """Test sensorimotor integration"""
        processor = EmbodiedCognitionProcessor(sensory_dim=512, motor_dim=128)
        
        sensory_input = torch.randn(512)
        motor_state = torch.randn(128)
        
        integrated, metrics = processor(sensory_input, motor_state)
        
        # Validate output shape
        assert integrated.shape == (512,)
        
        # Validate metrics
        assert 'sensorimotor_coupling' in metrics
        assert 'spatial_awareness' in metrics
        assert 'body_boundary' in metrics
        
        # Metrics should be in reasonable ranges
        assert -1.0 <= metrics['sensorimotor_coupling'] <= 1.0
        assert 0.0 <= metrics['spatial_awareness'] <= 2.0
        
    def test_body_schema_update(self):
        """Test body schema updates from feedback"""
        processor = EmbodiedCognitionProcessor()
        
        feedback = {
            'joint_1': 0.5,
            'joint_2': 0.8
        }
        
        processor.update_body_schema(feedback)
        
        assert 'joint_1' in processor.body_schema.joint_states
        assert processor.body_schema.joint_states['joint_1'] == 0.5
        
    def test_embodied_spatial_reasoning(self):
        """Test embodied spatial reasoning improves with motor state"""
        processor = EmbodiedCognitionProcessor()
        
        sensory = torch.randn(512)
        motor_active = torch.randn(128) * 2.0  # Strong motor signal
        motor_passive = torch.randn(128) * 0.1  # Weak motor signal
        
        _, metrics_active = processor(sensory, motor_active)
        _, metrics_passive = processor(sensory, motor_passive)
        
        # Active motor state should increase spatial awareness
        assert metrics_active['spatial_awareness'] > metrics_passive['spatial_awareness']


class TestEmbeddedCognition:
    """Test embedded cognition processor"""
    
    def test_embedded_initialization(self):
        """Test processor initializes correctly"""
        processor = EmbeddedCognitionProcessor(feature_dim=512, context_dim=256)
        assert processor.feature_dim == 512
        assert processor.context_dim == 256
        
    def test_embedded_processing(self):
        """Test context-aware processing"""
        processor = EmbeddedCognitionProcessor()
        
        features = torch.randn(512)
        context = {'temperature': 20.0, 'lighting': 'bright'}
        
        contextualized, metrics = processor(features, context)
        
        # Validate output
        assert contextualized.shape == (256,)  # context_dim
        
        # Validate metrics
        assert 'environmental_coupling' in metrics
        assert 'affordance_detection' in metrics
        assert 'context_stability' in metrics
        
    def test_affordance_detection(self):
        """Test affordance detection in environment"""
        processor = EmbeddedCognitionProcessor()
        
        features = torch.randn(512)
        context_embedding, _ = processor(features)
        
        affordances = processor.detect_affordances(context_embedding)
        
        # Should detect some affordances
        assert isinstance(affordances, list)
        assert all(aff.startswith('affordance_') for aff in affordances)
        
    def test_context_sensitive_adaptation(self):
        """Test behavior adapts to context"""
        processor = EmbeddedCognitionProcessor()
        
        features = torch.randn(512)
        
        # Process with different contexts
        context1 = {'type': 'indoor', 'obstacles': 'few'}
        context2 = {'type': 'outdoor', 'obstacles': 'many'}
        
        _, metrics1 = processor(features, context1)
        _, metrics2 = processor(features, context2)
        
        # Context should be reflected in processing
        assert processor.environmental_context.physical_features['type'] == 'outdoor'


class TestEnactedCognition:
    """Test enacted cognition processor"""
    
    def test_enacted_initialization(self):
        """Test processor initializes correctly"""
        processor = EnactedCognitionProcessor(feature_dim=512, action_dim=64)
        assert processor.feature_dim == 512
        assert processor.action_dim == 64
        
    def test_enacted_processing(self):
        """Test active perception with action"""
        processor = EnactedCognitionProcessor()
        
        features = torch.randn(512)
        action = torch.randn(64)
        
        predicted_state, metrics = processor(features, action)
        
        # Validate output
        assert predicted_state.shape == (512,)
        
        # Validate metrics
        assert 'exploration_intensity' in metrics
        assert 'prediction_confidence' in metrics
        
    def test_exploration_strategy(self):
        """Test autonomous exploration generation"""
        processor = EnactedCognitionProcessor()
        
        features = torch.randn(512)
        
        # Process without providing action - should generate exploration
        predicted_state, metrics = processor(features, action=None)
        
        assert predicted_state.shape == (512,)
        assert metrics['exploration_intensity'] > 0.0
        
    def test_action_outcome_learning(self):
        """Test learning from action-outcome pairs"""
        processor = EnactedCognitionProcessor()
        
        action = torch.randn(64)
        predicted = torch.randn(512)
        actual = torch.randn(512)
        
        initial_history_size = len(processor.action_history)
        
        processor.learn_action_outcome(action, predicted, actual, timestamp=1.0)
        
        # History should grow
        assert len(processor.action_history) == initial_history_size + 1
        
        # Should record prediction error
        assert processor.action_history[-1].prediction_error >= 0.0


class TestExtendedCognition:
    """Test extended cognition processor"""
    
    def test_extended_initialization(self):
        """Test processor initializes correctly"""
        processor = ExtendedCognitionProcessor(feature_dim=512, tool_dim=128)
        assert processor.feature_dim == 512
        assert processor.tool_dim == 128
        
    def test_extended_processing(self):
        """Test tool integration"""
        processor = ExtendedCognitionProcessor()
        
        cognitive_state = torch.randn(512)
        tool_state = torch.randn(128)
        
        extended, metrics = processor(cognitive_state, tool_state)
        
        # Validate output
        assert extended.shape == (512,)
        
        # Validate metrics
        assert 'tool_usage' in metrics
        assert 'cognitive_extension' in metrics
        assert 'tool_repertoire' in metrics
        
    def test_tool_registration(self):
        """Test registering new cognitive tools"""
        processor = ExtendedCognitionProcessor()
        
        tool_features = torch.randn(128)
        processor.register_tool('calculator', tool_features)
        
        assert 'calculator' in processor.tool_mastery
        assert processor.tool_mastery['calculator']['usage_count'] == 0
        
    def test_tool_use_and_mastery(self):
        """Test tool use increases mastery"""
        processor = ExtendedCognitionProcessor()
        
        tool_features = torch.randn(128)
        processor.register_tool('hammer', tool_features)
        
        initial_mastery = processor.tool_mastery['hammer']['mastery_level']
        
        # Use tool multiple times
        for _ in range(10):
            processor.use_tool('hammer')
        
        # Mastery should increase
        assert processor.tool_mastery['hammer']['mastery_level'] > initial_mastery
        assert processor.tool_mastery['hammer']['usage_count'] == 10
        
    def test_environmental_coupling(self):
        """Test cognitive extension to environment"""
        processor = ExtendedCognitionProcessor()
        
        cognitive_state = torch.randn(512)
        tool_state = torch.randn(128)
        
        extended, metrics = processor(cognitive_state, tool_state)
        
        # Extension should show coupling
        assert metrics['cognitive_extension'] != 0.0


class TestPerspectivalParticipatory:
    """Test perspectival and participatory knowing processor"""
    
    def test_perspectival_initialization(self):
        """Test processor initializes correctly"""
        processor = PerspectivalParticipatoryProcessor(feature_dim=512, perspective_dim=128)
        assert processor.feature_dim == 512
        assert processor.perspective_dim == 128
        
    def test_perspectival_processing(self):
        """Test perspective encoding"""
        processor = PerspectivalParticipatoryProcessor()
        
        features = torch.randn(512)
        
        perspectival_state, metrics = processor(features)
        
        # Validate output
        assert perspectival_state.shape == (128,)  # perspective_dim
        
        # Validate metrics
        assert 'perspective_flexibility' in metrics
        assert 'participation_depth' in metrics
        assert 'active_viewpoints' in metrics
        
    def test_multiple_viewpoint_integration(self):
        """Test integration of multiple perspectives"""
        processor = PerspectivalParticipatoryProcessor()
        
        features = torch.randn(512)
        context_perspective = torch.randn(128)
        
        integrated, metrics = processor(features, context_perspective)
        
        # Should integrate both perspectives
        assert integrated.shape == (128,)
        
    def test_perspective_flexibility(self):
        """Test flexibility in perspective shifting"""
        processor = PerspectivalParticipatoryProcessor()
        
        # Process multiple different perspectives
        for i in range(5):
            features = torch.randn(512) * (i + 1)  # Varying perspectives
            _, metrics = processor(features)
        
        # Flexibility should increase with diverse perspectives
        assert metrics['perspective_flexibility'] > 0.0
        assert metrics['active_viewpoints'] == 5


class TestProcessingModeManager:
    """Test processing mode management"""
    
    def test_mode_manager_initialization(self):
        """Test manager initializes correctly"""
        manager = ProcessingModeManager()
        assert manager.current_mode == ProcessingMode.CONFORMATIVE
        
    def test_conformative_mode_selection(self):
        """Test selection of conformative mode"""
        manager = ProcessingModeManager()
        
        # Low novelty, high coherence -> Conformative
        mode = manager.select_mode(novelty=0.3, coherence=0.8)
        
        assert mode == ProcessingMode.CONFORMATIVE
        
    def test_transformative_mode_selection(self):
        """Test selection of transformative mode"""
        manager = ProcessingModeManager()
        
        # High novelty, low coherence -> Transformative
        mode = manager.select_mode(novelty=0.9, coherence=0.3)
        
        assert mode == ProcessingMode.TRANSFORMATIVE
        
    def test_mode_statistics(self):
        """Test mode usage statistics"""
        manager = ProcessingModeManager()
        
        # Generate mixed mode history
        manager.select_mode(0.3, 0.8)  # Conformative
        manager.select_mode(0.3, 0.8)  # Conformative
        manager.select_mode(0.9, 0.3)  # Transformative
        
        stats = manager.get_mode_statistics()
        
        assert 'conformative_ratio' in stats
        assert 'transformative_ratio' in stats
        assert abs(stats['conformative_ratio'] - 2/3) < 0.01


class TestCognitiveEmotionalIntegration:
    """Test cognitive-emotional integration"""
    
    def test_integration_initialization(self):
        """Test integrator initializes correctly"""
        integrator = CognitiveEmotionalIntegrator(cognitive_dim=512, emotion_dim=64)
        assert integrator.cognitive_dim == 512
        assert integrator.emotion_dim == 64
        
    def test_integration_processing(self):
        """Test cognitive-emotional fusion"""
        integrator = CognitiveEmotionalIntegrator()
        
        cognitive_state = torch.randn(512)
        emotional_state = torch.randn(64)
        
        integrated, metrics = integrator(cognitive_state, emotional_state)
        
        # Validate output
        assert integrated.shape == (512,)
        
        # Validate metrics
        assert 'cognitive_emotional_synchrony' in metrics
        assert 'meaning_coherence' in metrics
        assert 'integration_quality' in metrics
        
    def test_meaning_coherence_evaluation(self):
        """Test meaning coherence computation"""
        integrator = CognitiveEmotionalIntegrator()
        
        cognitive = torch.randn(512)
        emotional = torch.randn(64)
        
        _, metrics = integrator(cognitive, emotional)
        
        # Coherence should be in valid range
        assert 0.0 <= metrics['meaning_coherence'] <= 1.0
        
    def test_enhanced_decision_making(self):
        """Test emotion enhances cognitive processing"""
        integrator = CognitiveEmotionalIntegrator()
        
        cognitive = torch.randn(512)
        emotional_strong = torch.randn(64) * 2.0
        emotional_weak = torch.randn(64) * 0.1
        
        _, metrics_strong = integrator(cognitive, emotional_strong)
        _, metrics_weak = integrator(cognitive, emotional_weak)
        
        # Integration quality should differ
        assert metrics_strong['integration_quality'] != metrics_weak['integration_quality']


class TestVervaeke4EFramework:
    """Test complete 4E framework integration"""
    
    def test_framework_initialization(self):
        """Test framework initializes all components"""
        config = {
            'feature_dim': 512,
            'motor_dim': 128,
            'context_dim': 256,
            'action_dim': 64,
            'tool_dim': 128,
            'perspective_dim': 128,
            'emotion_dim': 64
        }
        
        framework = Vervaeke4EFramework(config)
        
        assert framework.embodied is not None
        assert framework.embedded is not None
        assert framework.enacted is not None
        assert framework.extended is not None
        assert framework.perspectival_participatory is not None
        assert framework.cognitive_emotional is not None
        assert framework.mode_manager is not None
        
    def test_complete_cognitive_cycle(self):
        """Test complete 4E cognitive cycle"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        # Prepare inputs
        sensory_input = torch.randn(512)
        motor_state = torch.randn(128)
        emotional_state = torch.randn(64)
        context_info = {'location': 'indoor'}
        
        # Execute cycle
        final_state, all_metrics = framework.process_cycle(
            sensory_input=sensory_input,
            motor_state=motor_state,
            emotional_state=emotional_state,
            context_info=context_info
        )
        
        # Validate output
        assert final_state.shape == (512,)
        
        # Validate all component metrics present
        assert 'embodied' in all_metrics
        assert 'embedded' in all_metrics
        assert 'enacted' in all_metrics
        assert 'extended' in all_metrics
        assert 'perspectival' in all_metrics
        assert 'integration' in all_metrics
        assert 'processing_mode' in all_metrics
        assert 'framework' in all_metrics
        
    def test_framework_metrics(self):
        """Test framework-level metrics computation"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        
        _, metrics = framework.process_cycle(sensory, motor, emotion)
        
        # Validate framework metrics
        assert 'wisdom_measure' in metrics['framework']
        assert 'meaning_connectivity' in metrics['framework']
        assert 'overall_coherence' in metrics['framework']
        assert 'four_e_integration' in metrics['framework']
        
        # Metrics should be in valid ranges
        assert 0.0 <= metrics['framework']['wisdom_measure'] <= 1.5
        assert 0.0 <= metrics['framework']['meaning_connectivity'] <= 1.0
        
    def test_acceptance_criteria_embodied(self):
        """Validate: Embodied cognition improves sensorimotor integration"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor_active = torch.randn(128) * 2.0
        emotion = torch.randn(64)
        
        _, metrics = framework.process_cycle(sensory, motor_active, emotion)
        
        # Should show sensorimotor coupling
        assert metrics['embodied']['sensorimotor_coupling'] > -0.5
        assert metrics['embodied']['spatial_awareness'] > 0.0
        
    def test_acceptance_criteria_embedded(self):
        """Validate: Embedded cognition enables context-sensitive behavior"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        context = {'temperature': 25, 'complexity': 'high'}
        
        _, metrics = framework.process_cycle(sensory, motor, emotion, context_info=context)
        
        # Should show environmental coupling
        assert 'environmental_coupling' in metrics['embedded']
        assert metrics['embedded']['affordance_detection'] >= 0.0
        
    def test_acceptance_criteria_enacted(self):
        """Validate: Enacted cognition supports active perception"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        action = torch.randn(64)
        
        _, metrics = framework.process_cycle(sensory, motor, emotion, action=action)
        
        # Should show exploration behavior
        assert metrics['enacted']['exploration_intensity'] > 0.0
        assert 'prediction_confidence' in metrics['enacted']
        
    def test_acceptance_criteria_extended(self):
        """Validate: Extended cognition utilizes environmental affordances"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        tool = torch.randn(128)
        
        _, metrics = framework.process_cycle(sensory, motor, emotion, tool_state=tool)
        
        # Should show cognitive extension
        assert metrics['extended']['tool_usage'] > 0.0
        assert 'cognitive_extension' in metrics['extended']
        
    def test_acceptance_criteria_perspectival(self):
        """Validate: Perspectival knowing enables multiple viewpoint integration"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        # Process multiple cycles to build perspective history
        for i in range(3):
            sensory = torch.randn(512) * (i + 1)
            motor = torch.randn(128)
            emotion = torch.randn(64)
            
            _, metrics = framework.process_cycle(sensory, motor, emotion)
        
        # Should show perspective flexibility
        assert metrics['perspectival']['active_viewpoints'] > 0
        assert 'participation_depth' in metrics['perspectival']
        
    def test_acceptance_criteria_transformative(self):
        """Validate: Transformative processing facilitates insight"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        
        _, metrics = framework.process_cycle(sensory, motor, emotion)
        
        # Should select appropriate processing mode
        assert 'processing_mode' in metrics
        assert metrics['processing_mode'] in ['conformative', 'transformative']
        
    def test_acceptance_criteria_cognitive_emotional(self):
        """Validate: Cognitive-emotional integration enhances decision-making"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        
        _, metrics = framework.process_cycle(sensory, motor, emotion)
        
        # Should show meaningful integration
        assert metrics['integration']['meaning_coherence'] > 0.0
        assert metrics['integration']['cognitive_emotional_synchrony'] >= -1.0
        
    def test_wisdom_measure_computation(self):
        """Test wisdom measure integrates all 4E components"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        
        _, metrics = framework.process_cycle(sensory, motor, emotion)
        
        wisdom = metrics['framework']['wisdom_measure']
        
        # Wisdom should integrate embodied, embedded, enacted, extended
        assert wisdom > 0.0
        assert wisdom <= 1.5  # Reasonable upper bound
        
    def test_meaning_connectivity_anti_crisis(self):
        """Test meaning connectivity as anti-meaning-crisis measure"""
        config = {'feature_dim': 512}
        framework = Vervaeke4EFramework(config)
        
        sensory = torch.randn(512)
        motor = torch.randn(128)
        emotion = torch.randn(64)
        
        _, metrics = framework.process_cycle(sensory, motor, emotion)
        
        meaning = metrics['framework']['meaning_connectivity']
        
        # Should measure connection to meaning
        assert 0.0 <= meaning <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
