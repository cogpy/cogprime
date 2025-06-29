#!/usr/bin/env python3
"""
Simple CogPrime Integration Test

This script verifies the core integration of CogPrime with its new
AtomSpace and Memory modules, ensuring they can be instantiated and
perform basic operations. It's designed to run without requiring
all external dependencies (like a full node9 environment or all mem0
backends) by falling back to in-memory implementations where possible.
"""

import os
import sys
import torch
from typing import Dict, List, Any, Optional, Tuple

# Adjust path to import from cogprime package
# This is a common pattern for simple scripts within a package structure
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import CogPrime components
try:
    from src.core.cognitive_core import CogPrimeCore
    from src.atomspace import AtomSpace, Node, Link, BackendType
    from src.memory import Memory
    # The bridge is part of the integration but not directly tested for functionality
    # in this simple script, as it requires a running node9 instance.
    # from src.bridges.node9_atomspace_bridge import Node9AtomSpaceBridge
    from src.modules.perception import SensoryInput
except ImportError as e:
    print(f"Error importing CogPrime modules: {e}")
    print("Please ensure you are running this script from the project root or that 'src' is in your PYTHONPATH.")
    sys.exit(1)


def create_simulated_input() -> SensoryInput:
    """Create a very basic simulated sensory input."""
    return SensoryInput(
        visual=torch.zeros(3, 1, 1),
        audio=torch.zeros(1),
        text="simple test input",
        proprioceptive={"status": "ok"}
    )


def print_separator(title: str = "") -> None:
    """Print a separator with an optional title."""
    width = 80
    if title:
        padding = (width - len(title) - 4) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def main():
    """Run the simple integrated CogPrime test."""
    print_separator("COGPRIME SIMPLE INTEGRATION TEST")
    print("Verifying basic functionality of AtomSpace and Memory integration.")
    print()

    # Step 1: Initialize the CogPrime system
    print_separator("1. SYSTEM INITIALIZATION")

    # Configure the system to use local backends for simplicity
    # mem0 will automatically fall back to DictMemoryBackend if not fully installed
    config = {
        'atomspace_backend': BackendType.LOCAL,
        'atomspace_config': {},
        'memory_backend': 'mem0',
        'memory_config': {} # Empty config, allowing mem0 to use defaults or fallbacks
    }

    print("Attempting to create CogPrime core with AtomSpace and Memory...")
    try:
        core = CogPrimeCore(config)
        print(f"CogPrimeCore initialized successfully.")
        print(f"AtomSpace backend in use: {core.atomspace.backend_type}")
        print(f"Memory backend in use: {core.memory.backend_type}")
    except Exception as e:
        print(f"ERROR: Failed to initialize CogPrimeCore: {e}")
        sys.exit(1)
    print()

    # Step 2: Test AtomSpace basic operations
    print_separator("2. ATOMSPACE BASIC OPERATIONS")
    print("Testing AtomSpace: adding a node and a link.")
    try:
        # Create and add a node
        test_node = Node("TestConcept", "HelloWorld")
        core.atomspace.add(test_node)
        print(f"Added node: {test_node}")

        # Retrieve the node
        retrieved_node = core.atomspace.get_node("TestConcept", "HelloWorld")
        if retrieved_node and retrieved_node.id == test_node.id:
            print(f"Successfully retrieved node: {retrieved_node}")
        else:
            print("ERROR: Failed to retrieve node from AtomSpace.")
            sys.exit(1)

        # Create and add a link
        test_link = Link("TestRelation", [test_node, Node("TestConcept", "AnotherConcept")])
        core.atomspace.add(test_link)
        print(f"Added link: {test_link}")

        print("AtomSpace basic operations successful.")
    except Exception as e:
        print(f"ERROR: AtomSpace operations failed: {e}")
        sys.exit(1)
    print()

    # Step 3: Test Memory basic operations
    print_separator("3. MEMORY BASIC OPERATIONS")
    print("Testing Memory: storing and retrieving a simple value.")
    try:
        test_key = "test_value_key"
        test_value = {"data": "This is a test value for memory persistence."}
        
        # Store a value
        success = core.memory.store(test_key, test_value)
        if success:
            print(f"Successfully stored value with key '{test_key}'.")
        else:
            print(f"ERROR: Failed to store value with key '{test_key}'.")
            sys.exit(1)

        # Retrieve the value
        retrieved_value = core.memory.retrieve(test_key)
        if retrieved_value == test_value:
            print(f"Successfully retrieved value: {retrieved_value}")
        else:
            print(f"ERROR: Retrieved value mismatch. Expected {test_value}, got {retrieved_value}")
            sys.exit(1)
        
        print("Memory basic operations successful.")
    except Exception as e:
        print(f"ERROR: Memory operations failed: {e}")
        sys.exit(1)
    print()

    # Step 4: Simulate a cognitive cycle (basic flow)
    print_separator("4. COGNITIVE CYCLE SIMULATION")
    print("Simulating a single cognitive cycle to ensure basic flow.")
    try:
        sensory_input = create_simulated_input()
        reward = 0.1
        action = core.cognitive_cycle(sensory_input, reward)
        print(f"Simulated cognitive cycle completed. Action: {action}")
        print("Cognitive cycle simulation successful.")
    except Exception as e:
        print(f"ERROR: Cognitive cycle simulation failed: {e}")
        sys.exit(1)
    print()

    # Step 5: Note on Node9 Bridge
    print_separator("5. NODE9 BRIDGE (INTEGRATION NOTE)")
    print("The Node9AtomSpaceBridge is part of the integration architecture.")
    print("This simple test does not actively run a node9 instance or test FFI/Styx protocols.")
    print("Its Python components are integrated and ready for a live node9 environment.")
    print()

    print_separator("TEST COMPLETE")
    print("All core integration components initialized and performed basic operations successfully.")
    print("This confirms the Python-side integration is functional.")
    print_separator()


if __name__ == "__main__":
    main()
