#!/usr/bin/env python3
"""
Standalone test for Ennead-Based Relevance Realization Optimizer

This test runs independently without requiring torch or other heavy dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly without going through __init__.py
from core.ennead_relevance_optimizer import (
    EnneadRelevanceOptimizer,
    KnowingMode,
    UnderstandingOrder,
    WisdomPractice,
    EnneadState
)

def test_basic_functionality():
    """Test basic Ennead optimizer functionality"""
    print("=" * 70)
    print("Testing Ennead Relevance Realization Optimizer")
    print("=" * 70)
    
    # Test 1: Initialization
    print("\n[Test 1] Initializing optimizer...")
    optimizer = EnneadRelevanceOptimizer()
    assert optimizer is not None
    assert optimizer.state is not None
    print("✓ Optimizer initialized successfully")
    
    # Test 2: State structure
    print("\n[Test 2] Checking Ennead state structure...")
    state = optimizer.state
    assert len(state.knowing_dimensions) == 4, f"Expected 4 knowing dimensions, got {len(state.knowing_dimensions)}"
    assert len(state.order_dimensions) == 3, f"Expected 3 order dimensions, got {len(state.order_dimensions)}"
    assert len(state.wisdom_dimensions) == 3, f"Expected 3 wisdom dimensions, got {len(state.wisdom_dimensions)}"
    print("✓ Ennead structure validated: 4 + 3 + 3 = 10 dimensions")
    print(f"  - Knowing modes: {list(state.knowing_dimensions.keys())}")
    print(f"  - Understanding orders: {list(state.order_dimensions.keys())}")
    print(f"  - Wisdom practices: {list(state.wisdom_dimensions.keys())}")
    
    # Test 3: Basic relevance realization
    print("\n[Test 3] Testing basic relevance realization...")
    query = {'item_a', 'item_b', 'item_c'}
    context = {
        'facts': ['item_a is true'],
        'actions': ['item_b can be executed'],
        'salience_map': {'item_a': 0.8, 'item_b': 0.6, 'item_c': 0.4},
        'skill_level': 0.7,
        'virtue_alignment': 0.6
    }
    
    results = optimizer.realize_relevance(query, context)
    
    # Verify results structure
    assert 'integrated_relevance' in results
    assert 'triad_coherences' in results
    assert 'overall_integration' in results
    assert 'sophrosyne_score' in results
    
    print("✓ Relevance realization completed successfully")
    print(f"  - Overall integration: {results['overall_integration']:.3f}")
    print(f"  - Sophrosyne score: {results['sophrosyne_score']:.3f}")
    print(f"  - Triad coherences:")
    for triad, coherence in results['triad_coherences'].items():
        print(f"    • {triad}: {coherence:.3f}")
    
    # Test 4: Integrated relevance scores
    print("\n[Test 4] Examining integrated relevance scores...")
    integrated = results['integrated_relevance']
    print(f"  Found {len(integrated)} items with relevance scores:")
    for item, score in integrated.items():
        print(f"    • {item}: {score:.3f}")
    
    # Test 5: Emergent patterns
    print("\n[Test 5] Checking for emergent patterns...")
    patterns = results['emergent_patterns']
    print(f"  Detected {len(patterns)} emergent patterns:")
    for pattern in patterns:
        print(f"    • {pattern['type']}: strength={pattern['strength']:.3f}")
        if 'description' in pattern:
            print(f"      {pattern['description']}")
    
    # Test 6: Ways of knowing evaluation
    print("\n[Test 6] Evaluating through ways of knowing...")
    knowing_results = results['knowing_results']
    print("  Knowing mode activations:")
    for mode in KnowingMode:
        if mode.value in knowing_results:
            mode_data = knowing_results[mode.value]
            if isinstance(mode_data, dict) and mode_data:
                avg_relevance = sum(mode_data.values()) / len(mode_data)
                print(f"    • {mode.value}: avg relevance={avg_relevance:.3f}")
    print(f"  - Knowing integration: {knowing_results.get('knowing_integration', 0):.3f}")
    
    # Test 7: Orders of understanding
    print("\n[Test 7] Analyzing orders of understanding...")
    understanding_results = results['understanding_results']
    print("  Order coherences:")
    for order in UnderstandingOrder:
        if order.value in understanding_results:
            order_data = understanding_results[order.value]
            if isinstance(order_data, dict):
                print(f"    • {order.value}:")
                print(f"      - coherence: {order_data.get('coherence', 0):.3f}")
                print(f"      - salience: {order_data.get('salience', 0):.3f}")
                print(f"      - patterns: {len(order_data.get('patterns', []))}")
    print(f"  - Understanding integration: {understanding_results.get('understanding_integration', 0):.3f}")
    
    # Test 8: Wisdom practices
    print("\n[Test 8] Assessing wisdom practices...")
    wisdom_results = results['wisdom_results']
    print("  Wisdom cultivations:")
    for practice in WisdomPractice:
        if practice.value in wisdom_results:
            practice_data = wisdom_results[practice.value]
            if isinstance(practice_data, dict):
                print(f"    • {practice.value}:")
                print(f"      - cultivation: {practice_data.get('cultivation', 0):.3f}")
                print(f"      - alignment: {practice_data.get('alignment', 0):.3f}")
                metrics = practice_data.get('metrics', {})
                if metrics:
                    print(f"      - metrics: {', '.join(f'{k}={v:.2f}' for k, v in list(metrics.items())[:3])}")
    print(f"  - Wisdom integration: {wisdom_results.get('wisdom_integration', 0):.3f}")
    
    # Test 9: Sophrosyne optimization
    print("\n[Test 9] Examining sophrosyne optimization...")
    sophrosyne_opt = results.get('sophrosyne_optimization', {})
    print(f"  - Status: {sophrosyne_opt.get('status', 'unknown')}")
    if 'gradient' in sophrosyne_opt:
        print(f"  - Gradient: {sophrosyne_opt['gradient']:.3f}")
    adjustments = sophrosyne_opt.get('adjustments', [])
    if adjustments:
        print(f"  - Adjustments needed: {len(adjustments)}")
        for adj in adjustments[:3]:  # Show first 3
            print(f"    • {adj.get('triad', 'unknown')}: {adj.get('action', 'unknown')} by {adj.get('magnitude', 0):.3f}")
    else:
        print("  - No adjustments needed (optimal balance)")
    
    # Test 10: Multiple episodes for learning
    print("\n[Test 10] Testing learning through multiple episodes...")
    initial_integration = results['overall_integration']
    
    # Run 5 more episodes
    for i in range(5):
        optimizer.realize_relevance(query, context)
    
    final_results = optimizer.realize_relevance(query, context)
    final_integration = final_results['overall_integration']
    
    print(f"  - Initial integration: {initial_integration:.3f}")
    print(f"  - Final integration: {final_integration:.3f}")
    print(f"  - Change: {final_integration - initial_integration:+.3f}")
    print(f"  - Episodes recorded: {len(optimizer.relevance_episodes)}")
    
    # Test 11: State summary
    print("\n[Test 11] Getting comprehensive state summary...")
    summary = optimizer.get_state_summary()
    print("  State dimensions:")
    print(f"    • Knowing: {len(summary['knowing'])} dimensions")
    print(f"    • Understanding: {len(summary['understanding'])} dimensions")
    print(f"    • Wisdom: {len(summary['wisdom'])} dimensions")
    print(f"  Total episodes: {summary['episodes_recorded']}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    
    return True

def test_high_integration_scenario():
    """Test with context designed for high integration"""
    print("\n" + "=" * 70)
    print("Testing High Integration Scenario")
    print("=" * 70)
    
    optimizer = EnneadRelevanceOptimizer()
    
    query = {f'concept_{i}' for i in range(5)}
    context = {
        # High knowing activation
        'facts': [f'concept_{i} is true' for i in range(5)],
        'actions': [f'concept_{i} can be applied' for i in range(5)],
        'salience_map': {f'concept_{i}': 0.85 + i * 0.02 for i in range(5)},
        'identity_relevance': {f'concept_{i}': 0.8 for i in range(5)},
        'logical_coherence': 0.9,
        'procedural_fluency': 0.85,
        'aspect_perception': 0.88,
        'agapic_connection': 0.82,
        
        # High understanding coherence
        'causal_links': [{'from': f'concept_{i}', 'to': f'concept_{(i+1)%5}'} for i in range(5)],
        'values': [{'type': f'value_{i}', 'strength': 0.85} for i in range(5)],
        'temporal_sequences': [{'events': [f'concept_{i}' for i in range(5)]}],
        'expected_patterns': 3,
        'expected_values': 3,
        'expected_narrative': 1,
        'nomological_salience': 0.85,
        'normative_salience': 0.87,
        'narrative_salience': 0.83,
        
        # High wisdom cultivation
        'virtue_alignment': 0.85,
        'purpose_clarity': 0.88,
        'skill_level': 0.86,
        'flow_state': 0.84,
        'compassion_level': 0.87,
        'significance': 0.85,
        'performance_quality': 0.86,
        'connectedness': 0.84
    }
    
    results = optimizer.realize_relevance(query, context)
    
    print(f"\nResults with high integration context:")
    print(f"  - Overall integration: {results['overall_integration']:.3f}")
    print(f"  - Sophrosyne score: {results['sophrosyne_score']:.3f}")
    print(f"  - Emergent patterns: {len(results['emergent_patterns'])}")
    
    # Check for Ennead resonance
    resonance_patterns = [p for p in results['emergent_patterns'] if p['type'] == 'ennead_resonance']
    if resonance_patterns:
        print(f"\n✓ Ennead resonance detected!")
        for pattern in resonance_patterns:
            print(f"  - Strength: {pattern['strength']:.3f}")
            print(f"  - {pattern['description']}")
    else:
        print(f"\n  No full Ennead resonance (yet) - continue cultivation")
    
    print("\n" + "=" * 70)
    
    return True

if __name__ == '__main__':
    try:
        success = test_basic_functionality()
        if success:
            test_high_integration_scenario()
        print("\n✓ All Ennead optimization tests completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
