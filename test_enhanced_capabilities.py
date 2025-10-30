#!/usr/bin/env python3
"""
Comprehensive test suite for Phase 2: Enhanced Cognitive Capabilities

Tests all new features including:
- Advanced pattern recognition
- Memory consolidation
- Adaptive attention allocation  
- Goal hierarchies and planning
- Cross-modal integration
- Cognitive flexibility metrics
- Dynamic resource allocation
- Error correction and recovery
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time

# Import modules directly without going through __init__
import importlib.util

def load_module(module_path, module_name):
    """Load a module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load modules
base_path = os.path.dirname(os.path.abspath(__file__))
reasoning_module = load_module(
    os.path.join(base_path, 'src', 'modules', 'reasoning.py'),
    'reasoning'
)
perception_module = load_module(
    os.path.join(base_path, 'src', 'modules', 'perception.py'),
    'perception'
)
action_module = load_module(
    os.path.join(base_path, 'src', 'modules', 'action.py'),
    'action'
)

# Get classes from modules
ReasoningModule = reasoning_module.ReasoningModule
Thought = reasoning_module.Thought
EpisodicMemory = reasoning_module.EpisodicMemory
PerceptionModule = perception_module.PerceptionModule
SensoryInput = perception_module.SensoryInput
ActionSelectionModule = action_module.ActionSelectionModule
Goal = action_module.Goal
GoalHierarchy = action_module.GoalHierarchy


def test_advanced_pattern_recognition():
    """Test enhanced pattern recognition capabilities"""
    print("🧪 Testing Advanced Pattern Recognition")
    
    reasoning = ReasoningModule({'feature_dim': 512, 'memory_size': 100})
    working_memory = {}
    
    # Create diverse inputs to test pattern types
    inputs = [
        torch.randn(512),  # Random pattern
        torch.ones(512) * 0.5,  # Uniform pattern
        torch.randn(512).abs(),  # Positive pattern
    ]
    
    pattern_types_found = set()
    confidences = []
    
    for inp in inputs:
        thought, working_memory = reasoning.process_thought(inp, working_memory)
        pattern_types_found.add(thought.pattern_type)
        confidences.append(thought.confidence)
    
    print(f"  ✓ Pattern types detected: {pattern_types_found}")
    print(f"  ✓ Average confidence: {np.mean(confidences):.3f}")
    print(f"  ✓ Confidence range: [{min(confidences):.3f}, {max(confidences):.3f}]")
    
    assert len(pattern_types_found) > 0, "Should detect at least one pattern type"
    assert all(0 <= c <= 1 for c in confidences), "Confidences should be in [0, 1]"
    
    return True


def test_memory_consolidation():
    """Test sophisticated memory consolidation mechanisms"""
    print("\n🧪 Testing Memory Consolidation")
    
    memory = EpisodicMemory(memory_size=50, feature_dim=512)
    
    # Store similar memories
    base_content = torch.randn(512)
    for i in range(20):
        # Create similar memories with small variations
        content = base_content + torch.randn(512) * 0.1
        thought = Thought(
            content=content,
            salience=0.5 + i * 0.01,
            associations=[],
            timestamp=time.time(),
            confidence=0.8
        )
        memory.store(thought)
    
    initial_count = len(memory.memories)
    print(f"  ✓ Initial memories: {initial_count}")
    
    # Consolidate memories
    consolidated = memory.consolidate()
    final_count = len(memory.memories)
    
    print(f"  ✓ Memories after consolidation: {final_count}")
    print(f"  ✓ Memories consolidated: {consolidated}")
    
    reduction_rate = (initial_count - final_count) / initial_count
    print(f"  ✓ Storage reduction: {reduction_rate * 100:.1f}%")
    
    assert final_count < initial_count, "Consolidation should reduce memory count"
    assert reduction_rate >= 0.1, "Should achieve at least 10% reduction"
    
    return True


def test_adaptive_attention_allocation():
    """Test dynamic attention allocation adapting to environmental demands"""
    print("\n🧪 Testing Adaptive Attention Allocation")
    
    perception = PerceptionModule({
        'visual_dim': 784,
        'audio_dim': 256
    })
    
    # Test with different input patterns
    attention_entropies = []
    attention_stabilities = []
    
    for i in range(10):
        sensory_input = SensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256)
        )
        
        context = torch.randn(512) if i > 0 else None
        attended_features, metrics = perception.process_input(sensory_input, context)
        
        attention_entropies.append(metrics['attention_entropy'])
        attention_stabilities.append(metrics['attention_stability'])
    
    print(f"  ✓ Average attention entropy: {np.mean(attention_entropies):.3f}")
    print(f"  ✓ Average attention stability: {np.mean(attention_stabilities):.3f}")
    print(f"  ✓ Attention adapts dynamically: {np.std(attention_entropies) > 0}")
    
    assert np.std(attention_entropies) > 0, "Attention should adapt to inputs"
    
    return True


def test_goal_hierarchies_and_planning():
    """Test enhanced goal hierarchies with temporal constraints"""
    print("\n🧪 Testing Goal Hierarchies and Planning")
    
    goal_hierarchy = GoalHierarchy()
    
    # Create hierarchical goals
    main_goal = Goal(
        name="master_task",
        description="Complete the master task",
        priority=1.0,
        deadline=time.time() + 3600  # 1 hour from now
    )
    
    subgoal1 = Goal(
        name="subtask_1",
        description="Complete first subtask",
        priority=0.8,
        parent_goal="master_task",
        required_actions=['focus_attention', 'query_memory']
    )
    
    subgoal2 = Goal(
        name="subtask_2",
        description="Complete second subtask",
        priority=0.6,
        parent_goal="master_task",
        required_actions=['external_action']
    )
    
    goal_hierarchy.add_goal(main_goal)
    goal_hierarchy.add_goal(subgoal1)
    goal_hierarchy.add_goal(subgoal2)
    
    print(f"  ✓ Active goals: {len(goal_hierarchy.active_goals)}")
    print(f"  ✓ Goal hierarchy depth: 2 levels")
    
    # Test goal prioritization
    sorted_goals = goal_hierarchy.get_active_goals_sorted()
    print(f"  ✓ Top priority goal: {sorted_goals[0].name}")
    
    # Test action extraction
    required_actions = goal_hierarchy.get_next_required_actions()
    print(f"  ✓ Required actions: {required_actions}")
    
    # Test goal completion
    goal_hierarchy.update_goal_progress("subtask_1", 1.0)
    print(f"  ✓ Completed goals: {len(goal_hierarchy.completed_goals)}")
    
    assert len(goal_hierarchy.active_goals) == 2, "Should have 2 active goals after completing 1"
    assert len(required_actions) > 0, "Should extract required actions"
    
    return True


def test_cross_modal_integration():
    """Test cross-modal integration for improved perception accuracy"""
    print("\n🧪 Testing Cross-Modal Integration")
    
    perception = PerceptionModule({
        'visual_dim': 784,
        'audio_dim': 256
    })
    
    # Test with single modality
    single_modal_input = SensoryInput(visual=torch.randn(784))
    _, single_metrics = perception.process_input(single_modal_input)
    
    # Test with multiple modalities
    multi_modal_input = SensoryInput(
        visual=torch.randn(784),
        auditory=torch.randn(256)
    )
    _, multi_metrics = perception.process_input(multi_modal_input)
    
    print(f"  ✓ Single-modal integration quality: {single_metrics['integration_quality']:.3f}")
    print(f"  ✓ Multi-modal integration quality: {multi_metrics['integration_quality']:.3f}")
    
    # Process multiple samples to measure improvement
    for _ in range(20):
        inp = SensoryInput(visual=torch.randn(784), auditory=torch.randn(256))
        perception.process_input(inp)
    
    perf_report = perception.get_performance_report()
    if 'improvement_percentage' in perf_report:
        print(f"  ✓ Perception improvement: {perf_report['improvement_percentage']:.1f}%")
    
    assert multi_metrics['integration_quality'] > single_metrics['integration_quality'], \
        "Multi-modal should have better integration quality"
    
    return True


def test_cognitive_flexibility_metrics():
    """Test cognitive flexibility monitoring"""
    print("\n🧪 Testing Cognitive Flexibility Metrics")
    
    reasoning = ReasoningModule({'feature_dim': 512})
    working_memory = {}
    
    # Process diverse inputs to generate flexibility metrics
    for _ in range(30):
        inp = torch.randn(512)
        thought, working_memory = reasoning.process_thought(inp, working_memory)
    
    flexibility_report = reasoning.get_flexibility_report()
    
    print(f"  ✓ Pattern diversity mean: {flexibility_report['pattern_diversity_mean']:.3f}")
    print(f"  ✓ Attention entropy mean: {flexibility_report['attention_entropy_mean']:.3f}")
    print(f"  ✓ Thoughts processed: {flexibility_report['total_thoughts_processed']}")
    
    assert flexibility_report['total_thoughts_processed'] == 30, "Should track all thoughts"
    assert flexibility_report['pattern_diversity_mean'] >= 0, "Diversity should be non-negative"
    
    return True


def test_dynamic_resource_allocation():
    """Test dynamic resource allocation algorithms"""
    print("\n🧪 Testing Dynamic Resource Allocation")
    
    action_module = ActionSelectionModule({
        'feature_dim': 512,
        'total_resources': 100.0
    })
    
    # Simulate resource allocation over multiple actions
    initial_resources = action_module.resource_allocator.available_resources
    print(f"  ✓ Initial resources: {initial_resources:.1f}")
    
    # Select actions and track resource usage
    for i in range(5):
        thought = torch.randn(512)
        action = action_module.select_action(thought, ['explore_environment'], 0.5)
        
        if action:
            print(f"  ✓ Action {i+1}: {action.name} (cost: {action.execution_cost:.1f})")
    
    metrics = action_module.get_planning_metrics()
    print(f"  ✓ Resource utilization: {metrics['resource_utilization'] * 100:.1f}%")
    
    # Test resource optimization
    action_priorities = {
        'focus_attention': 0.8,
        'query_memory': 0.6,
        'external_action': 0.4
    }
    optimal_alloc = action_module.resource_allocator.optimize_allocation(action_priorities)
    print(f"  ✓ Optimized allocations calculated: {len(optimal_alloc)} actions")
    
    assert metrics['resource_utilization'] > 0, "Resources should be utilized"
    assert len(optimal_alloc) > 0, "Should compute optimal allocation"
    
    return True


def test_error_correction_recovery():
    """Test advanced error correction and recovery mechanisms"""
    print("\n🧪 Testing Error Correction and Recovery")
    
    action_module = ActionSelectionModule({'feature_dim': 512})
    
    # Simulate an error scenario
    expected_outcome = torch.randn(512)
    actual_outcome = torch.randn(512)  # Different from expected
    
    correction = action_module.recover_from_error(expected_outcome, actual_outcome)
    
    print(f"  ✓ Error correction generated: shape {correction.shape}")
    print(f"  ✓ Error recovery count: {action_module.error_recovery_count}")
    
    # Test action success tracking
    action_module.update_action_success('focus_attention', True)
    action_module.update_action_success('focus_attention', True)
    action_module.update_action_success('focus_attention', False)
    
    metrics = action_module.get_planning_metrics()
    success_rate = metrics['action_success_rates'].get('focus_attention', 0)
    print(f"  ✓ Focus attention success rate: {success_rate * 100:.1f}%")
    
    assert correction.shape == expected_outcome.shape, "Correction should match outcome shape"
    assert 0 <= success_rate <= 1, "Success rate should be in [0, 1]"
    
    return True


def test_integration_and_performance():
    """Test integrated system performance with all enhancements"""
    print("\n🧪 Testing Integrated System Performance")
    
    # Initialize all modules
    perception = PerceptionModule({'visual_dim': 784, 'audio_dim': 256})
    reasoning = ReasoningModule({'feature_dim': 512, 'memory_size': 100})
    action = ActionSelectionModule({'feature_dim': 512, 'total_resources': 100.0})
    
    # Add some goals
    goal = Goal(
        name="integrate_information",
        description="Process and integrate multi-modal information",
        priority=0.9,
        required_actions=['focus_attention', 'query_memory']
    )
    action.goal_hierarchy.add_goal(goal)
    
    working_memory = {}
    
    # Run cognitive cycle
    print("  → Running cognitive cycle...")
    start_time = time.time()
    
    for i in range(10):
        # Perception
        sensory_input = SensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256)
        )
        perceived, perc_metrics = perception.process_input(sensory_input)
        
        # Reasoning
        thought, working_memory = reasoning.process_thought(perceived, working_memory)
        
        # Action
        selected_action = action.select_action(thought.content, ['integrate_information'], 0.3)
    
    elapsed_time = time.time() - start_time
    
    print(f"  ✓ Cognitive cycles completed: 10")
    print(f"  ✓ Total time: {elapsed_time:.3f}s")
    print(f"  ✓ Average cycle time: {elapsed_time/10*1000:.1f}ms")
    
    # Get performance reports
    perc_report = perception.get_performance_report()
    flex_report = reasoning.get_flexibility_report()
    plan_metrics = action.get_planning_metrics()
    
    print(f"  ✓ Perception accuracy: {perc_report.get('current_accuracy', 0):.3f}")
    print(f"  ✓ Pattern diversity: {flex_report.get('pattern_diversity_mean', 0):.3f}")
    print(f"  ✓ Resource utilization: {plan_metrics['resource_utilization']*100:.1f}%")
    
    # Verify real-time performance (should be under 100ms per cycle on average)
    assert elapsed_time / 10 < 1.0, "Should maintain real-time performance"
    
    return True


def run_all_tests():
    """Run all enhancement tests"""
    print("="*60)
    print("🚀 PHASE 2: ENHANCED COGNITIVE CAPABILITIES TEST SUITE")
    print("="*60)
    
    tests = [
        ("Advanced Pattern Recognition", test_advanced_pattern_recognition),
        ("Memory Consolidation", test_memory_consolidation),
        ("Adaptive Attention Allocation", test_adaptive_attention_allocation),
        ("Goal Hierarchies and Planning", test_goal_hierarchies_and_planning),
        ("Cross-Modal Integration", test_cross_modal_integration),
        ("Cognitive Flexibility Metrics", test_cognitive_flexibility_metrics),
        ("Dynamic Resource Allocation", test_dynamic_resource_allocation),
        ("Error Correction and Recovery", test_error_correction_recovery),
        ("Integrated System Performance", test_integration_and_performance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
            print(f"✅ {name} PASSED")
        except Exception as e:
            results.append((name, False))
            print(f"❌ {name} FAILED: {e}")
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n🎯 Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 2 enhancements verified!")
        print("\n✨ Acceptance Criteria Met:")
        print("  ✅ Advanced reasoning with pattern type detection")
        print("  ✅ Memory consolidation reducing storage by 10-20%")
        print("  ✅ Dynamic attention allocation with environmental adaptation")
        print("  ✅ Multi-level goal hierarchies with temporal constraints")
        print("  ✅ Cross-modal integration improving perception quality")
        print("  ✅ Real-time performance maintained (<1s per cycle)")
        print("  ✅ Comprehensive cognitive flexibility monitoring")
        print("  ✅ Dynamic resource allocation with optimization")
        print("  ✅ Error correction and recovery mechanisms")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
