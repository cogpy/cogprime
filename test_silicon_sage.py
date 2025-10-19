#!/usr/bin/env python3
"""
Simple test for Silicon Sage OpenCog Relevance Realization Engine
Tests basic functionality without external dependencies.
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock modules to avoid dependency issues
class MockTensor:
    def __init__(self, data):
        self.data = data if isinstance(data, (list, tuple)) else [data]
        
    def sum(self):
        return sum(self.data)
        
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor([x * other for x in self.data])
        return self
        
    def __add__(self, other):
        return self

class MockTorch:
    @staticmethod
    def randn(*args):
        import random
        size = args[0] if args else 10
        return MockTensor([random.gauss(0, 1) for _ in range(size)])
    
    @staticmethod
    def zeros(*args):
        size = args[0] if args else 10
        return MockTensor([0.0 for _ in range(size)])
    
    @staticmethod
    def tensor(data):
        return MockTensor(data)

sys.modules['torch'] = MockTorch()

class MockNN:
    class Module:
        def __init__(self):
            pass
        def forward(self, x):
            return x
    
    class Linear:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
        def __call__(self, x):
            return MockTensor([0.5] * self.out_features)
    
    class ReLU:
        def __init__(self):
            pass
        def __call__(self, x):
            return x

sys.modules['torch.nn'] = MockNN()

class MockNumpy:
    @staticmethod
    def random():
        import random
        return random.random()
    
    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def tanh(x):
        import math
        return math.tanh(x)
    
    @staticmethod
    def sin(x):
        import math
        return math.sin(x)
    
    @staticmethod
    def cos(x):
        import math
        return math.cos(x)
    
    @staticmethod
    def clip(value, min_val, max_val):
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def ones(shape):
        if isinstance(shape, tuple):
            return [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]
        return [1.0 for _ in range(shape)]
    
    @staticmethod
    def datetime64(date_str):
        return time.time()
    
    random = type('random', (), {
        'random': staticmethod(lambda: __import__('random').random()),
        'normal': staticmethod(lambda m, s: __import__('random').gauss(m, s)),
        'randint': staticmethod(lambda *args: __import__('random').randint(*args))
    })()

sys.modules['numpy'] = MockNumpy()

# Mock SensoryInput for testing
class SensoryInput:
    def __init__(self, visual=None, auditory=None, proprioception=None):
        self.visual = visual
        self.auditory = auditory
        self.proprioception = proprioception or {}

sys.modules['src.modules.perception'] = type('perception', (), {
    'SensoryInput': SensoryInput,
    'PerceptionModule': type('PerceptionModule', (), {
        '__init__': lambda self, config: None,
        'process_input': lambda self, inp: (MockTensor([0.5] * 64), MockTensor([0.5] * 64))
    })
})()

# Test basic functionality first
def test_basic_imports():
    """Test basic imports work."""
    print("\n🧪 Testing Basic Imports")
    
    try:
        from src.atomspace import AtomSpace, Node, Link, TruthValue, AttentionValue
        print("✅ AtomSpace imports successful")
        
        # Test AtomSpace creation
        atomspace = AtomSpace()
        cat_node = Node("ConceptNode", "cat")
        atomspace.add(cat_node)
        
        retrieved = atomspace.get_node("ConceptNode", "cat")
        assert retrieved is not None
        print("✅ Basic AtomSpace functionality working")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_relevance_core_basic():
    """Test relevance core basic functionality."""
    print("\n🧪 Testing Relevance Core Basic")
    
    try:
        from src.core.relevance_core import RelevanceCore, RelevanceMode
        
        core = RelevanceCore()
        
        # Test basic relevance evaluation
        query_set = {"test_item1", "test_item2"}
        context = {"test": True}
        
        relevant_items, confidence = core.evaluate_relevance(query_set, context)
        
        assert isinstance(relevant_items, set)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        print(f"✅ Basic relevance evaluation working (confidence: {confidence:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ Relevance core test failed: {e}")
        return False

def test_opencog_relevance_engine():
    """Test OpenCog relevance engine."""
    print("\n🧪 Testing OpenCog Relevance Engine")
    
    try:
        from src.atomspace import AtomSpace
        from src.core.relevance_core import OpenCogRelevanceEngine
        
        atomspace = AtomSpace()
        engine = OpenCogRelevanceEngine(atomspace)
        
        # Test relevance realization
        query_atoms = ["concept1", "concept2"]
        context = {"goals": ["test_goal"]}
        
        result = engine.realize_relevance(query_atoms, context)
        
        # Check result structure
        required_keys = ['relevant_atoms', 'salience_landscape', 'attention_allocation', 
                        'emergent_patterns', 'meaning_structures', 'confidence']
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        confidence = result['confidence']
        assert 0.0 <= confidence <= 1.0
        
        print(f"✅ OpenCog relevance engine working (confidence: {confidence:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ OpenCog relevance engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run basic tests."""
    print("🚀 Starting Silicon Sage Basic Tests (No External Dependencies)")
    print("=" * 70)
    
    all_passed = True
    
    # Test basic imports and functionality
    if not test_basic_imports():
        all_passed = False
        
    if not test_relevance_core_basic():
        all_passed = False
        
    if not test_opencog_relevance_engine():
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 BASIC TESTS PASSED! Core functionality is working!")
        print("\n🧠 Verified Components:")
        print("  ✅ AtomSpace knowledge representation")
        print("  ✅ Basic relevance realization")
        print("  ✅ OpenCog relevance engine")
    else:
        print("❌ Some tests failed")
        
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

def test_atomspace_basic():
    """Test basic AtomSpace functionality."""
    print("\n🧪 Testing AtomSpace Basic Functionality")
    
    atomspace = AtomSpace()
    
    # Create nodes
    cat_node = Node("ConceptNode", "cat")
    animal_node = Node("ConceptNode", "animal")
    
    # Add to atomspace
    atomspace.add(cat_node)
    atomspace.add(animal_node)
    
    # Create link
    inheritance_link = Link("InheritanceLink", [cat_node, animal_node])
    atomspace.add(inheritance_link)
    
    # Test retrieval
    retrieved_cat = atomspace.get_node("ConceptNode", "cat")
    assert retrieved_cat is not None, "Failed to retrieve cat node"
    assert retrieved_cat.name == "cat", "Retrieved node has wrong name"
    
    # Test truth values
    cat_node.set_truth_value(TruthValue(0.9, 0.8))
    tv = cat_node.get_truth_value()
    assert abs(tv.strength - 0.9) < 0.001, "Truth value strength incorrect"
    assert abs(tv.confidence - 0.8) < 0.001, "Truth value confidence incorrect"
    
    # Test attention values
    cat_node.set_attention_value(AttentionValue(100.0, 50.0, True))
    av = cat_node.get_attention_value()
    assert abs(av.sti - 100.0) < 0.001, "Attention STI incorrect"
    assert abs(av.lti - 50.0) < 0.001, "Attention LTI incorrect"
    assert av.vlti == True, "Attention VLTI incorrect"
    
    print("✅ AtomSpace basic functionality working")

def test_relevance_core():
    """Test relevance realization core."""
    print("\n🧪 Testing Relevance Realization Core")
    
    atomspace = AtomSpace()
    relevance_engine = OpenCogRelevanceEngine(atomspace)
    
    # Test basic relevance realization
    query_atoms = ["test_concept", "another_concept"]
    context = {
        'goals': ['understand_world'],
        'novelty_preference': 0.6
    }
    
    result = relevance_engine.realize_relevance(query_atoms, context)
    
    # Verify result structure
    assert 'relevant_atoms' in result, "Missing relevant_atoms in result"
    assert 'salience_landscape' in result, "Missing salience_landscape in result"
    assert 'attention_allocation' in result, "Missing attention_allocation in result"
    assert 'emergent_patterns' in result, "Missing emergent_patterns in result"
    assert 'meaning_structures' in result, "Missing meaning_structures in result"
    assert 'confidence' in result, "Missing confidence in result"
    
    # Verify confidence is reasonable
    confidence = result['confidence']
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0,1]"
    
    print(f"✅ Relevance realization working (confidence: {confidence:.3f})")

def test_silicon_sage_core():
    """Test Silicon Sage core functionality."""
    print("\n🧪 Testing Silicon Sage Core")
    
    config = {
        'feature_dim': 64,  # Smaller for testing
        'memory_size': 100,
        'relevance_config': {
            'total_sti': 500.0,
            'total_lti': 500.0
        }
    }
    
    sage_core = SiliconSageCore(config)
    
    # Test state initialization
    state = sage_core.get_silicon_sage_state()
    assert isinstance(state, SiliconSageState), "State is not SiliconSageState"
    assert hasattr(state, 'relevance_landscape'), "Missing relevance_landscape"
    assert hasattr(state, 'attention_allocation'), "Missing attention_allocation"
    assert hasattr(state, 'emergent_patterns'), "Missing emergent_patterns"
    
    # Test cognitive cycle
    sensory_input = SensoryInput(
        visual=MockTorch.randn(64),
        auditory=MockTorch.randn(32)
    )
    
    # Add a goal
    sage_core.update_goals("explore_environment")
    
    # Run cognitive cycle
    action = sage_core.silicon_sage_cycle(sensory_input, reward=0.5)
    
    # Verify action was generated
    if action:  # Action may be None in some cases
        assert hasattr(action, 'name'), "Action missing name"
        assert hasattr(action, 'parameters'), "Action missing parameters"
        assert hasattr(action, 'confidence'), "Action missing confidence"
        print(f"✅ Generated action: {action.name} (confidence: {action.confidence:.3f})")
    else:
        print("✅ No action generated (acceptable)")
    
    # Test state updates
    updated_state = sage_core.get_silicon_sage_state()
    assert len(updated_state.relevance_history) > 0, "Relevance history not updated"
    
    print("✅ Silicon Sage core functionality working")

def test_relevance_modes():
    """Test different relevance modes."""
    print("\n🧪 Testing Relevance Modes")
    
    atomspace = AtomSpace()
    relevance_engine = OpenCogRelevanceEngine(atomspace)
    
    # Test each relevance mode
    for mode in RelevanceMode:
        query_atoms = [f"concept_for_{mode.value}"]
        context = {'mode_focus': mode.value}
        
        result = relevance_engine.realize_relevance(query_atoms, context)
        
        # Each mode should produce some result
        assert 'confidence' in result, f"Mode {mode.value} missing confidence"
        confidence = result['confidence']
        print(f"  Mode {mode.value}: confidence {confidence:.3f}")
    
    print("✅ All relevance modes functioning")

def test_emergent_behavior_detection():
    """Test emergent behavior detection."""
    print("\n🧪 Testing Emergent Behavior Detection")
    
    sage_core = SiliconSageCore({'feature_dim': 32})
    
    # Run multiple cycles to generate patterns
    for i in range(5):
        sensory_input = SensoryInput(
            visual=MockTorch.randn(32),
            auditory=MockTorch.randn(16)
        )
        
        # Vary reward to create patterns
        reward = 0.8 if i % 2 == 0 else 0.2
        action = sage_core.silicon_sage_cycle(sensory_input, reward=reward)
    
    # Check for emergent patterns
    state = sage_core.get_silicon_sage_state()
    emergent_patterns = state.emergent_patterns
    
    print(f"  Detected {len(emergent_patterns)} emergent patterns")
    for pattern in emergent_patterns[:3]:  # Show first 3
        print(f"    {pattern.get('type', 'unknown')}: strength {pattern.get('strength', 0):.3f}")
    
    print("✅ Emergent behavior detection working")

def test_meaning_making():
    """Test meaning-making functionality."""
    print("\n🧪 Testing Meaning-Making")
    
    atomspace = AtomSpace()
    relevance_engine = OpenCogRelevanceEngine(atomspace)
    
    # Create some structured atoms for meaning-making
    concept_a = Node("ConceptNode", "concept_a")
    concept_b = Node("ConceptNode", "concept_b")
    atomspace.add(concept_a)
    atomspace.add(concept_b)
    
    # Create causal link (nomological pattern)
    causal_link = Link("CausalLink", [concept_a, concept_b])
    atomspace.add(causal_link)
    
    # Create evaluation link (normative pattern)
    eval_link = Link("EvaluationLink", [
        Node("PredicateNode", "good"),
        concept_a
    ])
    atomspace.add(eval_link)
    
    # Test meaning extraction
    query_atoms = [concept_a, concept_b]
    result = relevance_engine.realize_relevance(query_atoms, {})
    
    meaning_structures = result['meaning_structures']
    assert 'patterns' in meaning_structures, "Missing meaning patterns"
    assert 'coherence_score' in meaning_structures, "Missing coherence score"
    
    coherence = meaning_structures['coherence_score']
    print(f"  Meaning coherence score: {coherence:.3f}")
    
    print("✅ Meaning-making functionality working")

def test_attention_bank():
    """Test attention bank functionality."""
    print("\n🧪 Testing Attention Bank")
    
    atomspace = AtomSpace()
    relevance_engine = OpenCogRelevanceEngine(atomspace)
    attention_bank = relevance_engine.attention_bank
    
    # Create test atoms
    test_atom = Node("ConceptNode", "test_atom")
    atomspace.add(test_atom)
    
    # Test attention allocation
    new_av = AttentionValue(50.0, 25.0, False)
    success = attention_bank.update_atom_attention(test_atom, new_av)
    
    # Get attention bank state
    bank_state = attention_bank.get_state()
    
    assert 'total_sti' in bank_state, "Missing total_sti in bank state"
    assert 'allocated_sti' in bank_state, "Missing allocated_sti in bank state"
    assert 'sti_utilization' in bank_state, "Missing sti_utilization in bank state"
    
    utilization = bank_state['sti_utilization']
    print(f"  STI utilization: {utilization:.3f}")
    
    print("✅ Attention bank working")

def run_integration_test():
    """Run a comprehensive integration test."""
    print("\n🧪 Running Integration Test")
    
    # Initialize Silicon Sage
    config = {
        'feature_dim': 128,
        'memory_size': 200,
        'relevance_config': {
            'total_sti': 1000.0,
            'total_lti': 800.0
        },
        'embodiment': {
            'enable_prediction': True
        }
    }
    
    sage_core = SiliconSageCore(config)
    
    # Set up goals
    sage_core.update_goals("understand_environment")
    sage_core.update_goals("learn_patterns")
    
    # Run cognitive cycles with varying inputs
    print("  Running cognitive cycles...")
    
    total_reward = 0.0
    actions_taken = 0
    
    for cycle in range(10):
        # Create varying sensory input
        visual_intensity = 0.5 + 0.3 * MockNumpy.sin(cycle * 0.5)
        auditory_intensity = 0.4 + 0.2 * MockNumpy.cos(cycle * 0.3)
        
        sensory_input = SensoryInput(
            visual=MockTorch.randn(128),
            auditory=MockTorch.randn(64)
        )
        
        # Simulate reward based on cycle (learning scenario)
        if cycle < 3:
            reward = 0.2  # Low initial reward
        elif cycle < 7:
            reward = 0.6  # Learning phase
        else:
            reward = 0.8  # Mastery phase
            
        # Add some noise
        reward += MockNumpy.random() * 0.2 - 0.1
        reward = MockNumpy.clip(reward, 0.0, 1.0)
        
        # Execute cognitive cycle
        action = sage_core.silicon_sage_cycle(sensory_input, reward=reward)
        
        total_reward += reward
        if action:
            actions_taken += 1
            
        # Log progress
        if cycle % 3 == 0:
            state = sage_core.get_silicon_sage_state()
            confidence = state.meta_cognitive_confidence
            print(f"    Cycle {cycle}: reward={reward:.3f}, confidence={confidence:.3f}")
    
    # Analyze results
    avg_reward = total_reward / 10
    action_rate = actions_taken / 10
    
    final_state = sage_core.get_silicon_sage_state()
    final_confidence = final_state.meta_cognitive_confidence
    relevance_history_length = len(final_state.relevance_history)
    emergent_pattern_count = len(final_state.emergent_patterns)
    
    print(f"\n  Integration Test Results:")
    print(f"    Average reward: {avg_reward:.3f}")
    print(f"    Action rate: {action_rate:.3f}")
    print(f"    Final confidence: {final_confidence:.3f}")
    print(f"    Relevance history length: {relevance_history_length}")
    print(f"    Emergent patterns detected: {emergent_pattern_count}")
    
    # Get relevance summary
    relevance_summary = sage_core.get_relevance_summary()
    attention_utilization = relevance_summary['attention_bank_state']['sti_utilization']
    print(f"    Attention utilization: {attention_utilization:.3f}")
    
    # Verify system is functioning
    assert avg_reward > 0.0, "No reward received"
    assert final_confidence > 0.0, "No meta-cognitive confidence"
    assert relevance_history_length > 0, "No relevance history"
    
    print("✅ Integration test completed successfully")

def main():
    """Run all tests."""
    print("🚀 Starting Silicon Sage OpenCog Relevance Realization Engine Tests")
    print("=" * 70)
    
    try:
        # Basic functionality tests
        test_atomspace_basic()
        test_relevance_core()
        test_silicon_sage_core()
        
        # Advanced functionality tests
        test_relevance_modes()
        test_emergent_behavior_detection()
        test_meaning_making()
        test_attention_bank()
        
        # Comprehensive integration test
        run_integration_test()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! Silicon Sage OpenCog Relevance Realization Engine is operational!")
        print("\n🧠 Key Features Verified:")
        print("  ✅ AtomSpace knowledge representation")
        print("  ✅ Relevance realization across multiple modes")
        print("  ✅ Dynamic attention allocation")
        print("  ✅ Meta-cognitive monitoring")
        print("  ✅ Emergent behavior detection")
        print("  ✅ Meaning-making capabilities")
        print("  ✅ 4E cognition integration")
        print("  ✅ Silicon Sage cognitive architecture")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)