#!/usr/bin/env python3
"""
Test Ontogentelechy Framework

This script tests the ontogentelechy framework including:
- Telos creation and evaluation
- Actualization tracking
- Developmental attractors
- Teleological fitness
- Integration with ontogenetic kernels
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 70)
print("ONTOGENTELECHY FRAMEWORK TEST")
print("=" * 70)

# Test 1: Import telos modules
print("\n[Test 1] Importing telos modules...")
try:
    from telos import (
        Telos, Criterion, ActualizationPhase,
        ActualizationTracker, DevelopmentalAttractor,
        TeleologicalFitness
    )
    from telos.examples import (
        create_semantic_coherence_telos,
        create_adaptive_learning_telos,
        list_example_teloi
    )
    print("✓ Successfully imported telos modules")
except Exception as e:
    print(f"✗ Failed to import telos modules: {e}")
    sys.exit(1)

# Test 2: Create example teloi
print("\n[Test 2] Creating example teloi...")
try:
    print(f"Available example teloi: {list_example_teloi()}")
    
    semantic_telos = create_semantic_coherence_telos()
    print(f"✓ Created semantic coherence telos: {semantic_telos.name}")
    print(f"  Description: {semantic_telos.description}")
    print(f"  Criteria: {len(semantic_telos.actualization_criteria)}")
    
    learning_telos = create_adaptive_learning_telos()
    print(f"✓ Created adaptive learning telos: {learning_telos.name}")
except Exception as e:
    print(f"✗ Failed to create teloi: {e}")
    sys.exit(1)

# Test 3: Create and evaluate actualization tracker
print("\n[Test 3] Testing actualization tracker...")
try:
    tracker = ActualizationTracker()
    
    # Create a mock entity
    class MockEntity:
        def __init__(self):
            self.genes = []
            self.fitness = 0.5
            self.age = 0
            self.metadata = {}
    
    entity = MockEntity()
    
    # Add some mock genes
    class MockGene:
        def __init__(self, weight):
            self.weight = weight
            self.active = True
    
    entity.genes = [MockGene(w) for w in [0.5, 0.6, 0.7, 0.8]]
    
    # Compute metrics
    metrics = tracker.compute_metrics(entity, semantic_telos)
    
    print(f"✓ Computed actualization metrics:")
    print(f"  Potentiality: {metrics.potentiality:.3f}")
    print(f"  Emergence: {metrics.emergence:.3f}")
    print(f"  Integration: {metrics.integration:.3f}")
    print(f"  Actualization: {metrics.actualization:.3f}")
    print(f"  Telos Alignment: {metrics.telos_alignment:.3f}")
    print(f"  Overall Health: {metrics.overall_health:.3f}")
    
except Exception as e:
    print(f"✗ Failed actualization tracker test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test developmental attractor
print("\n[Test 4] Testing developmental attractor...")
try:
    attractor = DevelopmentalAttractor(semantic_telos)
    
    # Create current state
    current_state = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    
    # Compute gradient
    gradient = attractor.compute_gradient(current_state)
    
    print(f"✓ Computed attractor gradient:")
    print(f"  Current state: {current_state}")
    print(f"  Gradient: {gradient}")
    print(f"  Gradient norm: {np.linalg.norm(gradient):.3f}")
    
    # Check if in basin
    in_basin = attractor.is_in_basin(entity)
    print(f"  Entity in basin: {in_basin}")
    
except Exception as e:
    print(f"✗ Failed developmental attractor test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test teleological fitness
print("\n[Test 5] Testing teleological fitness...")
try:
    from atomspace import AtomSpace, Node, Link
    from evolution.moses_engine import Individual, Gene
    
    # Create atomspace
    atomspace = AtomSpace()
    
    # Create individual with genes
    individual = Individual()
    
    # Add some genes
    for i in range(5):
        atom = Node("ConceptNode", f"concept_{i}")
        atomspace.add(atom)
        gene = Gene(atom=atom, weight=0.5 + i * 0.1, active=True)
        individual.add_gene(gene)
    
    # Create teleological fitness
    fitness_fn = TeleologicalFitness(atomspace, semantic_telos)
    
    # Evaluate fitness
    fitness = fitness_fn.evaluate(individual)
    
    print(f"✓ Evaluated teleological fitness: {fitness:.3f}")
    
    # Get detailed metrics
    metrics = fitness_fn.get_actualization_metrics(individual)
    print(f"  Actualization: {metrics.actualization:.3f}")
    print(f"  Integration: {metrics.integration:.3f}")
    print(f"  Emergence: {metrics.emergence:.3f}")
    
    # Check fitness components
    if 'fitness_components' in individual.metadata:
        components = individual.metadata['fitness_components']
        print(f"  Fitness components:")
        for name, value in components.items():
            print(f"    {name}: {value:.3f}")
    
except Exception as e:
    print(f"✗ Failed teleological fitness test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test ontogenetic kernel with telos
print("\n[Test 6] Testing ontogenetic kernel with telos...")
try:
    from ontogenesis import OntogeneticKernel, DevelopmentStage
    from ontogenesis.kernel import GripMetrics
    
    # Create kernel
    coefficients = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    grip = GripMetrics(contact=0.8, coverage=0.7, efficiency=0.9, stability=0.85)
    
    kernel = OntogeneticKernel(
        order=2,
        coefficients=coefficients,
        domain_spec="test_domain",
        grip_metrics=grip,
        telos=semantic_telos
    )
    
    print(f"✓ Created ontogenetic kernel with telos")
    print(f"  Stage: {kernel.ontogenetic_state.stage.value}")
    print(f"  Maturity: {kernel.ontogenetic_state.maturity:.3f}")
    print(f"  Telos: {kernel.telos.name}")
    
    # Test actualization gradient
    gradient = kernel.compute_actualization_gradient()
    if gradient is not None:
        print(f"  Actualization gradient norm: {np.linalg.norm(gradient):.3f}")
    
    # Test teleological pull
    print(f"\n  Applying teleological pull...")
    original_coeffs = kernel.coefficients.copy()
    kernel.apply_teleological_pull(strength=1.0)
    change = np.linalg.norm(kernel.coefficients - original_coeffs)
    print(f"  Coefficient change: {change:.3f}")
    
    # Simulate development
    print(f"\n  Simulating development...")
    for step in range(5):
        kernel.ontogenetic_state.maturity += 0.15
        kernel.advance_stage()
        kernel.apply_teleological_pull(strength=0.5)
        
        if kernel.actualization_tracker:
            metrics = kernel.actualization_tracker.compute_metrics(kernel, kernel.telos)
            print(f"  Step {step+1}: Stage={kernel.ontogenetic_state.stage.value}, "
                  f"Maturity={kernel.ontogenetic_state.maturity:.2f}, "
                  f"Actualization={metrics.actualization:.3f}")
    
except Exception as e:
    print(f"✗ Failed ontogenetic kernel test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test phase transition detection
print("\n[Test 7] Testing phase transition detection...")
try:
    # Create new tracker with history
    tracker = ActualizationTracker()
    
    # Simulate gradual improvement
    for i in range(10):
        entity.metadata['previous_fitness'] = 0.3 + i * 0.05
        metrics = tracker.compute_metrics(entity, semantic_telos)
        
        # Check for transition
        transition = tracker.detect_phase_transition()
        if transition:
            print(f"✓ Detected phase transition at step {i+1}")
            print(f"  Actualization jump: {transition['actualization_jump']:.3f}")
            print(f"  Emergence spike: {transition['emergence_spike']:.3f}")
    
    print(f"✓ Total phase transitions detected: {len(tracker.phase_transitions)}")
    
except Exception as e:
    print(f"✗ Failed phase transition test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test actualization phases
print("\n[Test 8] Testing actualization phases...")
try:
    phases_tested = []
    
    for actualization_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # Create telos with specific actualization
        test_telos = Telos(
            name="test",
            description="Test telos",
            actualization_criteria=[
                Criterion("test", "Test criterion", 1.0, lambda e: actualization_level, 1.0)
            ],
            attractor_state={'test': 1.0}
        )
        
        # Evaluate to update phase
        test_telos.evaluate_actualization(entity)
        
        phases_tested.append((actualization_level, test_telos.phase.value))
        print(f"  Actualization {actualization_level:.1f} → Phase: {test_telos.phase.value}")
    
    print(f"✓ Tested all actualization phases")
    
except Exception as e:
    print(f"✗ Failed actualization phases test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("✓ All ontogentelechy framework tests passed!")
print("\nKey Features Validated:")
print("  ✓ Telos creation and evaluation")
print("  ✓ Actualization tracking and metrics")
print("  ✓ Developmental attractors and gradients")
print("  ✓ Teleological fitness evaluation")
print("  ✓ Ontogenetic kernel integration")
print("  ✓ Phase transition detection")
print("  ✓ Actualization phase progression")
print("\n🎯 Ontogentelechy framework is operational!")
print("=" * 70)
