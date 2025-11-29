#!/usr/bin/env python3
"""
Visual Demonstration of Ennead Relevance Realization Optimization

This script demonstrates the Ennead optimizer in action with visual output.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.ennead_relevance_optimizer import EnneadRelevanceOptimizer
from core.ennead_integration_adapter import SimpleEnneadInterface

def print_header(text, width=70):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")

def print_bar(label, value, max_width=50):
    """Print a progress bar"""
    filled = int(value * max_width)
    bar = "█" * filled + "░" * (max_width - filled)
    print(f"{label:25s} [{bar}] {value:.3f}")

def demo_basic_optimization():
    """Demonstrate basic relevance optimization"""
    print_header("DEMO 1: Basic Relevance Optimization")
    
    optimizer = EnneadRelevanceOptimizer()
    
    # Define a cognitive task
    query = {'understand_concepts', 'apply_skills', 'realize_values'}
    context = {
        'facts': ['understand_concepts requires knowledge'],
        'actions': ['apply_skills in practice'],
        'salience_map': {
            'understand_concepts': 0.9,
            'apply_skills': 0.7,
            'realize_values': 0.8
        },
        'values': [{'type': 'wisdom', 'strength': 0.85}],
        'skill_level': 0.7,
        'virtue_alignment': 0.8
    }
    
    print("Context:")
    print(f"  Query: {query}")
    print(f"  Knowledge: {len(context.get('facts', []))} facts")
    print(f"  Skills: {context['skill_level']:.2f}")
    print(f"  Virtue: {context['virtue_alignment']:.2f}")
    
    # Realize relevance
    results = optimizer.realize_relevance(query, context)
    
    print("\n📊 Relevance Scores:")
    for item, score in sorted(
        results['integrated_relevance'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print_bar(item, score, max_width=40)
    
    print("\n🔱 Triad Coherences:")
    for triad, coherence in results['triad_coherences'].items():
        print_bar(triad.capitalize(), coherence)
    
    print("\n⚖️  Overall Metrics:")
    print_bar("Integration", results['overall_integration'])
    print_bar("Sophrosyne", results['sophrosyne_score'])
    
    if results['emergent_patterns']:
        print("\n✨ Emergent Patterns Detected:")
        for pattern in results['emergent_patterns']:
            print(f"  • {pattern['type']}: {pattern['strength']:.3f}")
            if 'description' in pattern:
                print(f"    {pattern['description']}")
    else:
        print("\n✨ No emergent patterns yet (need higher integration)")

def demo_learning_convergence():
    """Demonstrate learning and convergence"""
    print_header("DEMO 2: Learning and Convergence to Sophrosyne")
    
    optimizer = EnneadRelevanceOptimizer(config={
        'learning_rate': 0.05,
        'sophrosyne_target': 0.85
    })
    
    query = {'learning_item'}
    context = {
        'facts': ['learning_item is important'],
        'skill_level': 0.5,
        'virtue_alignment': 0.6
    }
    
    print("Running 15 optimization episodes...")
    print("\nEpisode | Integration | Sophrosyne | Pattern Count")
    print("-" * 60)
    
    integrations = []
    sophrosyne_scores = []
    
    for i in range(15):
        results = optimizer.realize_relevance(query, context)
        integration = results['overall_integration']
        sophrosyne = results['sophrosyne_score']
        patterns = len(results['emergent_patterns'])
        
        integrations.append(integration)
        sophrosyne_scores.append(sophrosyne)
        
        status = "✓" if integration > 0.75 else " "
        print(f"  {i+1:2d}    {status}  {integration:.3f}      {sophrosyne:.3f}        {patterns}")
    
    print("\n📈 Convergence Analysis:")
    early_int = sum(integrations[:5]) / 5
    late_int = sum(integrations[-5:]) / 5
    improvement = late_int - early_int
    
    print(f"  Early integration (ep 1-5):  {early_int:.3f}")
    print(f"  Late integration (ep 11-15): {late_int:.3f}")
    print(f"  Improvement:                 {improvement:+.3f}")
    
    if improvement > 0:
        print(f"  ✓ System converged toward optimal balance")
    else:
        print(f"  • System maintained stable integration")

def demo_high_integration():
    """Demonstrate high integration scenario"""
    print_header("DEMO 3: High Integration & Ennead Resonance")
    
    optimizer = EnneadRelevanceOptimizer()
    
    # Rich context for high integration
    query = {f'dimension_{i}' for i in range(5)}
    context = {
        # Strong knowing
        'facts': [f'dimension_{i} is understood' for i in range(5)],
        'actions': [f'apply_dimension_{i}' for i in range(5)],
        'salience_map': {f'dimension_{i}': 0.88 + i * 0.02 for i in range(5)},
        'identity_relevance': {f'dimension_{i}': 0.85 for i in range(5)},
        'logical_coherence': 0.92,
        'procedural_fluency': 0.88,
        'aspect_perception': 0.90,
        'agapic_connection': 0.85,
        
        # Strong understanding
        'causal_links': [
            {'from': f'dimension_{i}', 'to': f'dimension_{(i+1)%5}'}
            for i in range(5)
        ],
        'values': [{'type': f'value_{i}', 'strength': 0.88} for i in range(5)],
        'temporal_sequences': [{'events': [f'dimension_{i}' for i in range(5)]}],
        'expected_patterns': 3,
        'expected_values': 3,
        'expected_narrative': 1,
        'nomological_salience': 0.88,
        'normative_salience': 0.90,
        'narrative_salience': 0.86,
        
        # Strong wisdom
        'virtue_alignment': 0.88,
        'purpose_clarity': 0.92,
        'skill_level': 0.88,
        'flow_state': 0.86,
        'compassion_level': 0.90,
        'significance': 0.88,
        'performance_quality': 0.88,
        'connectedness': 0.86
    }
    
    print("Context designed for maximum integration:")
    print(f"  • Rich knowing: {len(context['facts'])} facts, {len(context['actions'])} actions")
    print(f"  • Deep understanding: {len(context['causal_links'])} causal links")
    print(f"  • Cultivated wisdom: high virtue (0.88), purpose (0.92), skill (0.88)")
    
    results = optimizer.realize_relevance(query, context)
    
    print("\n🌟 Integration Results:")
    print_bar("Overall Integration", results['overall_integration'])
    print_bar("Sophrosyne Score", results['sophrosyne_score'])
    
    print("\n🔱 Triad Coherences:")
    for triad, coherence in sorted(results['triad_coherences'].items()):
        status = "✓" if coherence > 0.80 else " "
        print(f"  {status} {triad.capitalize():15s} {coherence:.3f}")
    
    print("\n✨ Emergent Patterns:")
    if results['emergent_patterns']:
        for pattern in results['emergent_patterns']:
            icon = "🌟" if pattern['type'] == 'ennead_resonance' else "✦"
            print(f"  {icon} {pattern['type']}")
            print(f"     Strength: {pattern['strength']:.3f}")
            if 'description' in pattern:
                print(f"     {pattern['description']}")
    else:
        print("  None detected (need integration > 0.75 across all triads)")
    
    # Check for full resonance
    resonance = [p for p in results['emergent_patterns'] if p['type'] == 'ennead_resonance']
    if resonance:
        print("\n🎊 ENNEAD RESONANCE ACHIEVED! 🎊")
        print("   All 10 dimensions are fully integrated and working in harmony.")
        print(f"   Resonance strength: {resonance[0]['strength']:.3f}")

def demo_simple_interface():
    """Demonstrate simple interface"""
    print_header("DEMO 4: Simple Interface for Easy Usage")
    
    interface = SimpleEnneadInterface()
    
    items = ['learn', 'practice', 'teach', 'innovate', 'reflect']
    context = {
        'goals': ['mastery', 'wisdom'],
        'values': [{'type': 'excellence', 'strength': 0.9}]
    }
    
    print("Finding relevant items for goals: mastery, wisdom")
    print(f"Items: {items}\n")
    
    relevant = interface.find_relevant(items, context)
    
    print("📋 Relevance Ranking:")
    for rank, (item, score) in enumerate(relevant, 1):
        stars = "★" * int(score * 5)
        print(f"  {rank}. {item:12s} {stars:10s} {score:.3f}")
    
    print(f"\n📊 System State:")
    integration = interface.get_integration_level()
    wisdom = interface.get_wisdom_balance()
    
    print_bar("Integration Level", integration)
    print("\n  Wisdom Balance:")
    for practice, level in wisdom.items():
        print(f"    {practice:12s} {level:.3f}")

def main():
    """Run all demonstrations"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " ENNEAD RELEVANCE REALIZATION OPTIMIZER ".center(68) + "║")
    print("║" + " Visual Demonstration ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    try:
        demo_basic_optimization()
        input("\n[Press Enter to continue to Demo 2...]")
        
        demo_learning_convergence()
        input("\n[Press Enter to continue to Demo 3...]")
        
        demo_high_integration()
        input("\n[Press Enter to continue to Demo 4...]")
        
        demo_simple_interface()
        
        print_header("Demonstration Complete!")
        print("The Ennead Relevance Optimizer successfully demonstrated:")
        print("  ✓ Multi-dimensional relevance evaluation (10 dimensions)")
        print("  ✓ Learning and convergence to sophrosyne")
        print("  ✓ Emergent pattern detection (including full resonance)")
        print("  ✓ Simple interface for practical usage")
        print("\nFor more information, see:")
        print("  • docs/ENNEAD_RELEVANCE_OPTIMIZER.md")
        print("  • RELEVANCE_OPTIMIZATION_SUMMARY.md")
        print()
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
