# OpenCog in Pure Limbo

This directory contains a pure Limbo implementation of core OpenCog components, providing an alternative to the Python and Lua implementations.

## Overview

Limbo is a statically-typed programming language from Bell Labs' Inferno OS. This implementation provides:

- **AtomSpace**: Hypergraph knowledge representation
- **Perception Module**: Sensory processing and attention allocation
- **Reasoning Module**: Pattern recognition, working memory, and episodic memory
- **Action Module**: Behavior generation and goal-directed planning
- **Learning Module**: Experience-based adaptation and skill refinement
- **Cognitive Core**: Integrated cognitive cycle combining all modules

## Architecture

### AtomSpace (`atomspace.b`, `atomspace.m`)

The AtomSpace is the core knowledge representation system, implementing a hypergraph database where:
- **Nodes** represent concepts, percepts, and entities
- **Links** represent relationships and predicates
- **TruthValues** represent probabilistic truth (strength and confidence)
- **AttentionValues** represent cognitive resource allocation (STI, LTI, VLTI)

Key features:
- Efficient indexing by ID, type, and type-name pairs
- Pattern matching and query capabilities
- Event-driven architecture for module integration

### Cognitive Modules

1. **Perception** (`perception.b`, `perception.m`)
   - Multi-modal sensory processing (visual, auditory, tactile)
   - Attention allocation based on salience
   - Attention spreading through hypergraph
   - Sensory encoding into perception nodes

2. **Reasoning** (`reasoning.b`, `reasoning.m`)
   - Working memory with capacity limits
   - Attention-based activation
   - Pattern recognition and matching
   - Episodic memory for event storage
   - Inference capabilities (simplified PLN)

3. **Action** (`action.b`, `action.m`)
   - Action representation with parameters
   - Goal-directed behavior
   - Epsilon-greedy action selection
   - Expected reward evaluation

4. **Learning** (`learning.b`, `learning.m`)
   - Experience buffer for reinforcement learning
   - Skill acquisition and refinement
   - Truth value updates based on experience
   - Pattern consolidation from experiences

### Cognitive Core (`cogcore.b`, `cogcore.m`)

Integrates all modules into a unified cognitive cycle:

1. **Perception**: Process sensory input → percepts
2. **Reasoning**: Update working memory, recognize patterns
3. **Action Selection**: Choose action based on state and goals
4. **Learning**: Update from experience, evaluate actions
5. **Attention**: Allocate and spread attention
6. **Consolidation**: Periodic knowledge consolidation

## File Structure

```
limbo/
├── opencog/             # Core OpenCog implementation
│   ├── atomspace.b      # AtomSpace implementation
│   ├── atomspace.m      # AtomSpace module interface
│   ├── perception.b     # Perception module implementation
│   ├── perception.m     # Perception module interface
│   ├── reasoning.b      # Reasoning module implementation
│   ├── reasoning.m      # Reasoning module interface
│   ├── action.b         # Action module implementation
│   ├── action.m         # Action module interface
│   ├── learning.b       # Learning module implementation
│   ├── learning.m       # Learning module interface
│   ├── cogcore.b        # Cognitive core implementation
│   └── cogcore.m        # Cognitive core interface
├── examples/            # Example programs
│   └── simple_agent.b   # Simple cognitive agent example
├── tests/               # Unit tests
│   └── test_atomspace.b # AtomSpace module tests
├── README.md            # This file
├── Makefile             # Build system
└── build.sh             # Build script
```

## Building

To compile the Limbo modules (requires Inferno or compatible Limbo compiler):

```bash
# Compile modules
limbo opencog/atomspace.b
limbo opencog/perception.b
limbo opencog/reasoning.b
limbo opencog/action.b
limbo opencog/learning.b
limbo opencog/cogcore.b

# Compile examples
cd examples
limbo simple_agent.b
```

## Usage Example

```limbo
implement SimpleAgent;

include "sys.m";
include "cogcore.m";
    cogcore: CogCore;
    CogCore, CogConfig: import cogcore;
include "perception.m";
    perception: Perception;
    SensoryInput: import perception;

init(nil: ref Draw->Context, args: list of string)
{
    sys = load Sys Sys->PATH;
    cogcore = load CogCore CogCore->PATH;
    perception = load Perception Perception->PATH;
    
    # Create cognitive core with default configuration
    config := CogConfig.default();
    core := CogCore.mk(config);
    
    # Create some sensory input
    visual := array[784] of real;
    for (i := 0; i < len visual; i++)
        visual[i] = 0.5;
    
    audio := array[256] of real;
    for (i := 0; i < len audio; i++)
        audio[i] = 0.1;
    
    input := SensoryInput.mk(visual, audio, nil);
    
    # Run cognitive cycle
    for (cycle := 0; cycle < 100; cycle++) {
        reward := 1.0;  # Positive reward
        action := core.cognitive_cycle(input, reward);
        
        if (action != nil) {
            sys->print("Cycle %d: Selected action: %s\n", 
                      cycle, action.name);
        }
        
        # Print stats every 10 cycles
        if (cycle % 10 == 0) {
            stats := core.get_stats();
            sys->print("%s\n", stats);
        }
    }
}
```

## Comparison with Python Implementation

### Advantages of Limbo Implementation

1. **Static Typing**: Compile-time type checking prevents many runtime errors
2. **Memory Safety**: Automatic garbage collection without manual memory management
3. **Modularity**: Clean module system with explicit interfaces
4. **Performance**: Compiled to DIS bytecode, faster than interpreted Python
5. **Distributed**: Native support for 9P protocol and distributed computing
6. **Concurrency**: Built-in support for concurrent processes via channels

### Differences from Python Implementation

1. **No Dynamic Typing**: More verbose but safer type declarations
2. **Explicit Memory**: References (ref) must be explicitly declared
3. **List-based Collections**: Uses Limbo's list types instead of Python lists/dicts
4. **No Object Inheritance**: Uses ADTs (abstract data types) instead
5. **Functional Style**: More functional programming patterns

## Integration with Node9

The Limbo implementation can integrate with Node9 (Lua-based) through:

1. **9P Protocol**: Expose AtomSpace as a 9P filesystem
2. **Message Channels**: Inter-process communication
3. **FFI Bindings**: Foreign function interface for Lua↔Limbo calls

See the Python `node9_atomspace_bridge.py` for bridge patterns.

## Design Principles

Following OpenCog principles:

1. **Hypergraph Representation**: Knowledge as atoms and links
2. **Probabilistic Truth Values**: Uncertain reasoning
3. **Attention Allocation**: Limited cognitive resources
4. **Cognitive Synergy**: Integration of multiple learning/reasoning approaches
5. **Emergence**: Complex behaviors from simple module interactions

## Future Enhancements

- [ ] Full PLN (Probabilistic Logic Networks) implementation
- [ ] ECAN (Economic Attention Network) for attention allocation
- [ ] Pattern mining algorithms for learning
- [ ] Integration with MeTTa language
- [ ] Distributed AtomSpace across 9P network
- [ ] GPU acceleration for pattern matching
- [ ] More sophisticated inference rules
- [ ] Visual debugging tools

## References

- [Limbo Programming Language](http://www.vitanuova.com/inferno/limbo.html)
- [OpenCog Hyperon](https://github.com/trueagi-io/hyperon)
- [CogPrime Architecture](../docs/ARCHITECTURE.md)
- [Inferno Operating System](http://www.vitanuova.com/inferno/)

## License

MIT License - see LICENSE file for details.
