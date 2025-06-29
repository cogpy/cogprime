#!/usr/bin/env python3
"""
Integrated CogPrime Demo

This script demonstrates the integration of CogPrime with:
- AtomSpace for distributed knowledge representation
- mem0 for persistent memory and vector search
- node9 for distributed namespace and process management

The demo shows a complete cognitive cycle with:
1. Knowledge representation in AtomSpace
2. Cognitive processing with simulated sensory input
3. Memory operations with semantic search and fact extraction
4. Distributed operation via node9 namespace
5. Persistence and recovery of cognitive state
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple

# Import CogPrime components
from cogprime.core.cognitive_core import CogPrimeCore
from cogprime.atomspace import AtomSpace, Node, Link, BackendType
from cogprime.memory import Memory
from cogprime.bridges.node9_atomspace_bridge import Node9AtomSpaceBridge
from cogprime.modules.perception import SensoryInput


def create_simulated_input(complexity: float = 0.5) -> SensoryInput:
    """
    Generate simulated sensory input data with adjustable complexity for cognitive system testing.
    
    Parameters:
        complexity (float): A value from 0.0 (simplest) to 1.0 (most complex) that scales the richness and challenge of the generated input.
    
    Returns:
        SensoryInput: An object containing random visual (RGB image), audio (waveform), selected text, and proprioceptive (body state) data reflecting the specified complexity.
    """
    # Create visual input (simulated image tensor)
    visual = torch.randn(3, 64, 64)  # RGB image, 64x64
    
    # Create audio input (simulated audio waveform)
    audio = torch.randn(16000)  # 1 second at 16kHz
    
    # Create textual input (simulated text)
    text_options = [
        "The cat is sleeping on the windowsill.",
        "There's a bird flying outside the window.",
        "I can hear someone talking in the next room.",
        "The room temperature is comfortable.",
        "I need to solve this complex problem."
    ]
    text = text_options[int(complexity * (len(text_options) - 1))]
    
    # Create proprioceptive input (simulated body state)
    proprioceptive = {
        "position": [0.5, 0.3, 0.7],  # x, y, z coordinates
        "orientation": [0.0, 0.0, 0.0, 1.0],  # quaternion
        "energy": 0.8 - (complexity * 0.3),  # energy level decreases with complexity
        "stress": complexity * 0.7  # stress increases with complexity
    }
    
    return SensoryInput(
        visual=visual,
        audio=audio,
        text=text,
        proprioceptive=proprioceptive
    )


def print_separator(title: str = "") -> None:
    """
    Prints a horizontal separator line of 80 characters, optionally centering a provided title within the line.
    """
    width = 80
    if title:
        padding = (width - len(title) - 4) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def print_atom(atom, indent: int = 0) -> None:
    """
    Recursively prints the structure of an AtomSpace atom with indentation for hierarchical visualization.
    
    Parameters:
        atom: The AtomSpace atom to print.
        indent (int): The current indentation level for nested atoms.
    """
    indent_str = "  " * indent
    if atom.is_node():
        print(f"{indent_str}{atom.atom_type}(\"{atom.name}\")")
    else:
        print(f"{indent_str}{atom.atom_type}(")
        for outgoing in atom.outgoing_set:
            print_atom(outgoing, indent + 1)
        print(f"{indent_str})")


def main():
    """
    Runs a full demonstration of the integrated CogPrime cognitive architecture, showcasing knowledge representation with AtomSpace, memory operations with mem0, cognitive cycles with simulated sensory input, distributed namespace integration via node9, state persistence and recovery, and final system cleanup.
    
    The demo proceeds stepwise through system initialization, creation and querying of knowledge, memory fact management and semantic search, execution of cognitive cycles with varying input complexity, simulated distributed operation, saving and restoring cognitive state, and reporting of final statistics.
    """
    print_separator("COGPRIME INTEGRATED DEMO")
    print("Demonstrating the integration of CogPrime with AtomSpace, mem0, and node9")
    print()
    
    # Step 1: Initialize the CogPrime system
    print_separator("1. SYSTEM INITIALIZATION")
    
    # Configure the system
    config = {
        'atomspace_backend': BackendType.LOCAL,  # Start with local backend for demo
        'atomspace_config': {},
        'memory_backend': 'mem0',  # Use mem0 for memory
        'memory_config': {
            'vector_store': {
                'provider': 'chroma',
                'config': {
                    'collection_name': 'cogprime_demo'
                }
            },
            'enable_graph': True  # Enable graph memory
        }
    }
    
    # Create the CogPrime core
    print("Creating CogPrime core with AtomSpace and mem0...")
    core = CogPrimeCore(config)
    
    print(f"AtomSpace backend: {core.atomspace.backend_type}")
    print(f"Memory backend: {core.memory.backend_type}")
    print("Core system initialized successfully")
    print()
    
    # Step 2: Knowledge representation with AtomSpace
    print_separator("2. KNOWLEDGE REPRESENTATION WITH ATOMSPACE")
    
    # Create some basic knowledge in the AtomSpace
    print("Creating knowledge in AtomSpace...")
    
    # Create concept nodes
    concept_human = Node("ConceptNode", "human")
    concept_animal = Node("ConceptNode", "animal")
    concept_cat = Node("ConceptNode", "cat")
    concept_dog = Node("ConceptNode", "dog")
    concept_intelligence = Node("ConceptNode", "intelligence")
    
    # Add to AtomSpace
    core.atomspace.add(concept_human)
    core.atomspace.add(concept_animal)
    core.atomspace.add(concept_cat)
    core.atomspace.add(concept_dog)
    core.atomspace.add(concept_intelligence)
    
    # Create relationships
    inheritance_human_animal = Link("InheritanceLink", [concept_human, concept_animal])
    inheritance_cat_animal = Link("InheritanceLink", [concept_cat, concept_animal])
    inheritance_dog_animal = Link("InheritanceLink", [concept_dog, concept_animal])
    
    # Add to AtomSpace
    core.atomspace.add(inheritance_human_animal)
    core.atomspace.add(inheritance_cat_animal)
    core.atomspace.add(inheritance_dog_animal)
    
    # Create evaluation links
    predicate_has = Node("PredicateNode", "has")
    evaluation_human_intelligence = Link("EvaluationLink", [
        predicate_has,
        Link("ListLink", [concept_human, concept_intelligence])
    ])
    
    # Add to AtomSpace
    core.atomspace.add(predicate_has)
    core.atomspace.add(evaluation_human_intelligence)
    
    print("Knowledge added to AtomSpace")
    print()
    
    # Query the AtomSpace
    print("Querying AtomSpace for animals:")
    
    # Create a pattern to match all animals
    pattern = Link("InheritanceLink", [None, concept_animal])
    results = core.atomspace.query(pattern)
    
    print(f"Found {len(results)} animal concepts:")
    for result in results:
        print_atom(result)
    print()
    
    # Step 3: Memory operations with mem0
    print_separator("3. MEMORY OPERATIONS WITH MEM0")
    
    # Add facts to memory
    print("Adding facts to memory...")
    facts = [
        "Cats are carnivorous mammals with retractable claws.",
        "Dogs are domesticated descendants of wolves.",
        "Humans have developed complex language and technology.",
        "Animals are multicellular organisms that form the biological kingdom Animalia.",
        "Intelligence is the ability to acquire and apply knowledge and skills."
    ]
    
    fact_ids = []
    for i, fact in enumerate(facts):
        fact_id = core.memory.add_fact(fact, tags=[f"fact_{i}"])
        fact_ids.append(fact_id)
        print(f"Added fact: {fact}")
    
    print()
    
    # Extract facts from text
    print("Extracting facts from text using LLM...")
    text = """
    The brain is the central organ of the nervous system in all vertebrates.
    It consists of approximately 86 billion neurons and an equal number of glial cells.
    The human brain weighs about 1.5 kilograms and consumes about 20% of the body's oxygen.
    """
    
    try:
        extracted_facts = core.memory.extract_facts(text)
        print(f"Extracted {len(extracted_facts)} facts:")
        for i, fact in enumerate(extracted_facts):
            print(f"  {i+1}. {fact.get('fact', '')}")
    except Exception as e:
        print(f"Fact extraction not available: {e}")
    
    print()
    
    # Semantic search
    print("Performing semantic search...")
    query = "What do we know about intelligent mammals?"
    
    try:
        results = core.memory.semantic_search(query, limit=3)
        print(f"Top {len(results)} results for query: '{query}'")
        for i, (key, value, score) in enumerate(results):
            content = value.get("content", value) if isinstance(value, dict) else value
            print(f"  {i+1}. [{score:.4f}] {content}")
    except Exception as e:
        print(f"Semantic search not available: {e}")
    
    print()
    
    # Step 4: Cognitive cycles with simulated input
    print_separator("4. COGNITIVE CYCLES WITH SIMULATED INPUT")
    
    # Run cognitive cycles with increasing complexity
    print("Running cognitive cycles with simulated sensory input...")
    
    for i in range(5):
        complexity = i / 4.0  # 0.0 to 1.0
        sensory_input = create_simulated_input(complexity)
        reward = 0.5 - (abs(0.5 - complexity) * 0.5)  # Reward peaks at medium complexity
        
        print(f"\nCycle {i+1} (complexity: {complexity:.2f}, reward: {reward:.2f}):")
        print(f"  Text input: {sensory_input.text}")
        
        # Execute cognitive cycle
        action = core.cognitive_cycle(sensory_input, reward)
        
        # Print results
        print(f"  Attention focus: {core.state.attention_focus.sum().item():.4f}")
        print(f"  Emotional valence: {core.state.emotional_valence:.4f}")
        print(f"  Current thought: {core.state.current_thought}")
        print(f"  Action: {action}")
        
        # Add knowledge based on the cycle
        if i == 2:  # Add special knowledge on the middle cycle
            # Create a concept for the current thought
            thought_text = str(core.state.current_thought)[:50]
            thought_node = Node("ThoughtNode", f"thought_{i}")
            content_node = Node("ConceptNode", thought_text)
            
            # Create a link between the thought and its content
            thought_link = Link("EvaluationLink", [
                Node("PredicateNode", "has_content"),
                Link("ListLink", [thought_node, content_node])
            ])
            
            # Add to AtomSpace
            core.atomspace.add(thought_node)
            core.atomspace.add(content_node)
            core.atomspace.add(thought_link)
            
            print(f"  Added thought to AtomSpace: {thought_text}")
    
    print()
    
    # Step 5: Node9 namespace integration
    print_separator("5. NODE9 NAMESPACE INTEGRATION")
    
    print("Setting up node9 bridge for distributed operation...")
    
    try:
        # Create the bridge
        bridge = Node9AtomSpaceBridge(core.atomspace, "/cog")
        
        # Start the bridge (simulated for the demo)
        print("Bridge initialized (simulation mode)")
        print("AtomSpace now accessible through node9 namespace at /cog/atoms/")
        
        # Register cognitive modules with the bridge
        bridge.register_cognitive_module("perception", "cogprime.modules.perception")
        bridge.register_cognitive_module("reasoning", "cogprime.modules.reasoning")
        bridge.register_cognitive_module("action", "cogprime.modules.action")
        bridge.register_cognitive_module("learning", "cogprime.modules.learning")
        
        print("Registered cognitive modules as virtual processes:")
        for module in ["perception", "reasoning", "action", "learning"]:
            print(f"  - {module} at /cog/modules/{module}/")
        
        print("\nSimulating node9 filesystem operations:")
        print("  $ cat /cog/atoms/types/ConceptNode")
        print("  ConceptNode")
        print("  $ cat /cog/control/status")
        print("  running")
        print("  $ echo '{\"type\":\"ConceptNode\",\"name\":\"example\"}' > /cog/queries/add_node")
        print("  $ cat /cog/atoms/[new_id]")
        print("  {\"id\":\"...\",\"type\":\"ConceptNode\",\"name\":\"example\",\"tv\":{...}}")
    except Exception as e:
        print(f"Bridge simulation error: {e}")
    
    print()
    
    # Step 6: Persistence and recovery
    print_separator("6. PERSISTENCE AND RECOVERY")
    
    # Save the cognitive state
    print("Saving cognitive state...")
    save_path = "demo_state"
    success = core.save_state(save_path)
    print(f"State saved to {save_path}: {success}")
    
    # Modify the state
    original_valence = core.state.emotional_valence
    core.state.emotional_valence = 0.0
    print(f"Modified emotional valence: {original_valence:.4f} -> {core.state.emotional_valence:.4f}")
    
    # Recover the state
    print("Recovering cognitive state...")
    success = core.load_state(save_path)
    print(f"State recovered: {success}")
    print(f"Recovered emotional valence: {core.state.emotional_valence:.4f}")
    
    # Verify recovery
    if abs(core.state.emotional_valence - original_valence) < 0.001:
        print("Recovery successful! State matches the original.")
    else:
        print("Recovery issue: state does not match the original.")
    
    print()
    
    # Step 7: Final status and cleanup
    print_separator("7. FINAL STATUS AND CLEANUP")
    
    # Print final statistics
    print("Final system statistics:")
    print(f"  Total reward accumulated: {core.state.total_reward:.4f}")
    print(f"  Working memory size: {len(core.state.working_memory)} items")
    print(f"  Goal stack: {core.state.goal_stack}")
    
    # Clean up resources
    print("\nCleaning up resources...")
    
    # Close the bridge
    try:
        bridge.stop()
        print("Node9 bridge stopped")
    except:
        pass
    
    print("\nDemo completed successfully!")
    print_separator()


if __name__ == "__main__":
    main()
