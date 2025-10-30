# OpenCog Limbo Implementation Guide

## Overview

This guide provides comprehensive documentation for the OpenCog implementation in pure Limbo, a statically-typed programming language from Bell Labs' Inferno operating system.

## Table of Contents

1. [Architecture](#architecture)
2. [Modules](#modules)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Integration](#integration)
6. [Development](#development)

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Cognitive Core                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Perception │→→│  Reasoning │→→│   Action   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│         ↓              ↓                ↓                   │
│  ┌──────────────────────────────────────────────┐          │
│  │            AtomSpace (Hypergraph)             │          │
│  └──────────────────────────────────────────────┘          │
│         ↑                                                   │
│  ┌────────────┐                                            │
│  │  Learning  │                                            │
│  └────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Cognitive Cycle

1. **Perception Phase**: Sensory input → Perception nodes
2. **Attention Phase**: Allocate STI, spread attention
3. **Reasoning Phase**: Pattern matching, working memory update
4. **Action Phase**: Select and execute action
5. **Learning Phase**: Update from experience, consolidate

## Modules

### AtomSpace Module

**File**: `opencog/atomspace.b`, `opencog/atomspace.m`

Core hypergraph knowledge representation.

#### Key Types

```limbo
Atom: adt {
    id:         string;
    atomtype:   string;
    tv:         ref TruthValue;
    av:         ref AttentionValue;
    values:     list of (string, string);
};

Node: adt {
    atom:       Atom;
    name:       string;
    mk:         fn(atomtype: string, name: string): ref Node;
};

Link: adt {
    atom:       Atom;
    outgoing:   array of ref Atom;
    mk:         fn(atomtype: string, outgoing: array of ref Atom): ref Link;
};

AtomSpace: adt {
    mk:         fn(): ref AtomSpace;
    add_node:   fn(space: self ref AtomSpace, atomtype: string, name: string): ref Node;
    add_link:   fn(space: self ref AtomSpace, atomtype: string, outgoing: array of ref Atom): ref Link;
    query:      fn(space: self ref AtomSpace, pattern: ref Atom): list of ref Atom;
};
```

#### Common Atom Types

- **ConceptNode**: Represents concepts (e.g., "cat", "animal")
- **PredicateNode**: Represents predicates/properties
- **PerceptionNode**: Sensory percepts
- **InheritanceLink**: A is-a B relationships
- **EvaluationLink**: Predicate evaluations
- **ExecutionLink**: Procedural knowledge

### Perception Module

**File**: `opencog/perception.b`, `opencog/perception.m`

Handles sensory processing and attention allocation.

#### Key Types

```limbo
SensoryInput: adt {
    visual:     array of real;
    auditory:   array of real;
    tactile:    array of real;
    mk:         fn(visual, auditory, tactile: array of real): ref SensoryInput;
};

PerceptionModule: adt {
    mk:         fn(space: ref AtomSpace, visual_dim: int, audio_dim: int): ref PerceptionModule;
    process:    fn(pm: self ref PerceptionModule, input: ref SensoryInput): list of ref Node;
    encode:     fn(pm: self ref PerceptionModule, data: array of real, modality: string): ref Node;
};
```

### Reasoning Module

**File**: `opencog/reasoning.b`, `opencog/reasoning.m`

Pattern recognition, inference, and memory management.

#### Key Types

```limbo
WorkingMemoryItem: adt {
    atom:       ref Atom;
    activation: real;
    timestamp:  int;
};

ReasoningModule: adt {
    mk:         fn(space: ref AtomSpace, wm_capacity: int, attention_decay: real): ref ReasoningModule;
    infer:      fn(rm: self ref ReasoningModule, premises: list of ref Atom): list of ref Link;
    recognize_pattern: fn(rm: self ref ReasoningModule, pattern: ref Atom): list of ref Atom;
};
```

### Action Module

**File**: `opencog/action.b`, `opencog/action.m`

Action selection and goal-directed behavior.

#### Key Types

```limbo
Action: adt {
    name:       string;
    params:     list of real;
    expected_reward: real;
    mk:         fn(name: string, params: list of real): ref Action;
};

ActionModule: adt {
    mk:         fn(space: ref AtomSpace, exploration_rate: real): ref ActionModule;
    select_action: fn(am: self ref ActionModule, state: list of ref Node): ref Action;
};
```

### Learning Module

**File**: `opencog/learning.b`, `opencog/learning.m`

Experience-based learning and skill acquisition.

#### Key Types

```limbo
Experience: adt {
    state:      list of ref Node;
    action:     string;
    reward:     real;
    next_state: list of ref Node;
};

LearningModule: adt {
    mk:         fn(space: ref AtomSpace, learning_rate: real, buffer_size: int): ref LearningModule;
    learn_from_experience: fn(lm: self ref LearningModule, exp: ref Experience): int;
};
```

### Cognitive Core Module

**File**: `opencog/cogcore.b`, `opencog/cogcore.m`

Integrated cognitive architecture.

#### Key Types

```limbo
CogConfig: adt {
    visual_dim:     int;
    audio_dim:      int;
    wm_capacity:    int;
    learning_rate:  real;
    mk:             fn(): ref CogConfig;
};

CogCore: adt {
    mk:             fn(config: ref CogConfig): ref CogCore;
    cognitive_cycle: fn(core: self ref CogCore, input: ref SensoryInput, reward: real): ref Action;
    get_stats:      fn(core: self ref CogCore): string;
};
```

## API Reference

### Creating an AtomSpace

```limbo
space := AtomSpace.mk();
```

### Adding Nodes

```limbo
# Create a concept node
cat := space.add_node("ConceptNode", "cat");
animal := space.add_node("ConceptNode", "animal");

# Set truth value
tv := TruthValue.mk(0.9, 0.8);
cat.atom.tv = tv;
```

### Creating Links

```limbo
# Create inheritance link: cat -> animal
outgoing := array[2] of ref Atom;
outgoing[0] = ref cat.atom;
outgoing[1] = ref animal.atom;
link := space.add_link("InheritanceLink", outgoing);
```

### Querying

```limbo
# Find all ConceptNodes
concepts := space.get_atoms_by_type("ConceptNode");

# Pattern matching
pattern := space.add_node("ConceptNode", "cat");
matches := space.query(ref pattern.atom);
```

### Running Cognitive Cycle

```limbo
# Create configuration
config := CogConfig.mk();

# Create cognitive core
core := CogCore.mk(config);

# Create sensory input
visual := array[784] of real;
audio := array[256] of real;
# ... initialize arrays ...
input := SensoryInput.mk(visual, audio, nil);

# Run cycle
reward := 1.0;
action := core.cognitive_cycle(input, reward);
```

## Usage Examples

### Example 1: Basic Knowledge Representation

```limbo
space := AtomSpace.mk();

# Create nodes
dog := space.add_node("ConceptNode", "dog");
mammal := space.add_node("ConceptNode", "mammal");
pet := space.add_node("ConceptNode", "pet");

# Create relationships
outgoing1 := array[2] of ref Atom;
outgoing1[0] = ref dog.atom;
outgoing1[1] = ref mammal.atom;
is_mammal := space.add_link("InheritanceLink", outgoing1);

outgoing2 := array[2] of ref Atom;
outgoing2[0] = ref dog.atom;
outgoing2[1] = ref pet.atom;
is_pet := space.add_link("InheritanceLink", outgoing2);
```

### Example 2: Cognitive Agent

See `examples/simple_agent.b` for a complete cognitive agent implementation.

## Integration

### Integration with Python

Use the existing `node9_atomspace_bridge.py` as a reference for bridging:

1. **Serialize atoms to JSON** for inter-language communication
2. **Use 9P protocol** for distributed access
3. **Message channels** for asynchronous communication

### Integration with Node9/Lua

1. **FFI bindings**: Create C wrapper for Limbo functions
2. **9P filesystem**: Expose AtomSpace as virtual files
3. **Channels**: Use Inferno message channels

## Development

### Building

Requires Limbo compiler (Inferno or compatible):

```bash
limbo opencog/atomspace.b
limbo opencog/perception.b
limbo opencog/reasoning.b
limbo opencog/action.b
limbo opencog/learning.b
limbo opencog/cogcore.b
```

### Testing

Create test programs following the pattern in `examples/simple_agent.b`:

```limbo
implement TestAtomSpace;

include "atomspace.m";

init(nil: ref Draw->Context, args: list of string)
{
    space := AtomSpace.mk();
    
    # Test node creation
    node := space.add_node("ConceptNode", "test");
    if (node == nil)
        sys->print("FAIL: Could not create node\n");
    else
        sys->print("PASS: Node created\n");
    
    # Test size
    if (space.size() != 1)
        sys->print("FAIL: Size should be 1\n");
    else
        sys->print("PASS: Size is correct\n");
}
```

### Best Practices

1. **Always check for nil**: Limbo doesn't have exceptions
2. **Use ref for heap objects**: Automatic garbage collection
3. **Explicit module loading**: Check load results
4. **Type safety**: Leverage static typing
5. **Memory efficiency**: Use lists for collections

### Performance Optimization

1. **Indexing**: AtomSpace uses multiple indices for fast lookup
2. **List operations**: Functional lists are efficient in Limbo
3. **Avoid copies**: Use references where possible
4. **Batch operations**: Process multiple atoms together

## Future Enhancements

1. **Pattern Matcher**: Full pattern matching engine
2. **PLN**: Probabilistic Logic Networks implementation  
3. **ECAN**: Economic Attention Network
4. **MeTTa Integration**: Meta-type system
5. **Distributed AtomSpace**: Multi-node 9P network
6. **Persistence**: Database backend
7. **Visualization**: Debug and monitoring tools

## Contributing

Contributions welcome! Areas of focus:

- Performance optimization
- Additional cognitive modules
- Better pattern matching
- Integration with other systems
- Documentation improvements
- Test coverage

## References

- [Limbo Language Reference](http://www.vitanuova.com/inferno/papers/limbo.html)
- [Inferno Programming](http://www.vitanuova.com/inferno/papers/)
- [OpenCog Architecture](https://wiki.opencog.org/w/The_Open_Cognition_Project)
- [CogPrime Book](http://goertzel.org/CogPrime_Overview_Paper.pdf)

## License

MIT License - See LICENSE file for details.
