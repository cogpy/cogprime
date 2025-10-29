# OpenCog in Pure Lua

A complete implementation of the OpenCog cognitive architecture in pure Lua, featuring AtomSpace hypergraph knowledge representation, probabilistic logic networks (PLN), attention allocation, pattern matching, and learning capabilities.

## Features

- **AtomSpace**: Hypergraph-based knowledge representation system
- **Nodes & Links**: Fundamental building blocks for knowledge graphs
- **TruthValue**: Probabilistic truth values with strength and confidence
- **AttentionValue**: Economic attention allocation with STI/LTI
- **Pattern Matcher**: Advanced pattern matching with wildcards and variable binding
- **Attention Bank**: ECAN (Economic Attention Network) implementation
- **PLN Reasoner**: Probabilistic Logic Networks with deduction, induction, and abduction
- **Learning**: Pattern mining, reinforcement learning, and concept formation

## Installation

### Prerequisites

- Lua 5.3 or higher

### Setup

```bash
# Install Lua (Ubuntu/Debian)
sudo apt-get install lua5.3

# Clone the repository
git clone https://github.com/drzo/cogprime.git
cd cogprime/lua
```

## Quick Start

### Basic Usage

```lua
local opencog = require('opencog.init')

-- Create an AtomSpace
local atomspace = opencog.create_atomspace()

-- Create nodes
local cat = opencog.create_node("ConceptNode", "cat")
local animal = opencog.create_node("ConceptNode", "animal")

-- Create a link
local inheritance = opencog.create_link("InheritanceLink", {cat, animal})

-- Set truth value
inheritance:set_truth_value(opencog.TruthValue.new(0.9, 0.8))

-- Add to atomspace
atomspace:add(cat)
atomspace:add(animal)
atomspace:add(inheritance)

print("AtomSpace size:", atomspace:size())
```

### Pattern Matching

```lua
local opencog = require('opencog.init')
local atomspace = opencog.create_atomspace()

-- Add some knowledge
local cat = opencog.create_node("ConceptNode", "cat")
local dog = opencog.create_node("ConceptNode", "dog")
local animal = opencog.create_node("ConceptNode", "animal")

atomspace:add(cat)
atomspace:add(dog)
atomspace:add(animal)
atomspace:add(opencog.create_link("InheritanceLink", {cat, animal}))
atomspace:add(opencog.create_link("InheritanceLink", {dog, animal}))

-- Query with wildcards (empty outgoing set matches all)
local pattern = opencog.create_link("InheritanceLink", {})
local matches = atomspace:query(pattern)

print("Found", #matches, "inheritance links")
```

### Reasoning with PLN

```lua
local opencog = require('opencog.init')
local atomspace = opencog.create_atomspace()
local reasoner = opencog.Reasoner.new(atomspace)

-- Create knowledge: cat -> mammal -> animal
local cat = opencog.create_node("ConceptNode", "cat")
local mammal = opencog.create_node("ConceptNode", "mammal")
local animal = opencog.create_node("ConceptNode", "animal")

local cat_mammal = opencog.create_link("InheritanceLink", {cat, mammal})
local mammal_animal = opencog.create_link("InheritanceLink", {mammal, animal})

cat_mammal:set_truth_value(opencog.TruthValue.new(0.95, 0.9))
mammal_animal:set_truth_value(opencog.TruthValue.new(1.0, 1.0))

-- Perform deduction: cat -> animal
local inference = reasoner:deduction(cat_mammal, mammal_animal)
print("Deduced:", tostring(inference))
print("Truth value:", tostring(inference:get_truth_value()))
```

### Attention Allocation

```lua
local opencog = require('opencog.init')
local atomspace = opencog.create_atomspace()
local attention_bank = opencog.AttentionBank.new(atomspace, {
    total_sti = 1000.0,
    sti_threshold = 50.0
})

-- Add atoms
local important = opencog.create_node("ConceptNode", "important")
atomspace:add(important)

-- Stimulate important atoms
attention_bank:stimulate(important, 100)

-- Update attention dynamics
attention_bank:update_attention()

-- Get focused atoms
local focused = attention_bank:get_attentional_focus()
print("Atoms in focus:", #focused)
```

### Learning

```lua
local opencog = require('opencog.init')
local atomspace = opencog.create_atomspace()
local learning = opencog.Learning.new(atomspace)

-- Add some knowledge
local cat = opencog.create_node("ConceptNode", "cat")
local pet = opencog.create_node("ConceptNode", "pet")
atomspace:add(cat)
atomspace:add(pet)
atomspace:add(opencog.create_link("InheritanceLink", {cat, pet}))

-- Mine patterns
local patterns = learning:mine_frequent_patterns(1)
for _, pattern in ipairs(patterns) do
    print("Pattern:", pattern.pattern, "Support:", pattern.support)
end

-- Reinforce knowledge
learning:reinforce_atom(cat, 50)
```

## Architecture

### Core Components

#### AtomSpace
The AtomSpace is a hypergraph database that stores all knowledge as atoms (nodes and links).

- **Nodes**: Represent concepts, predicates, or entities
- **Links**: Represent relationships between atoms

#### TruthValue
Represents probabilistic truth with:
- **Strength**: Probability (0.0 to 1.0)
- **Confidence**: Certainty in the probability (0.0 to 1.0)

#### AttentionValue
Economic attention allocation with:
- **STI**: Short-Term Importance (dynamic)
- **LTI**: Long-Term Importance (stable)
- **VLTI**: Very Long-Term Importance flag

### Advanced Components

#### Pattern Matcher
- Wildcard matching with empty outgoing sets
- Variable binding with substitution
- Constraint-based filtering

#### Reasoner (PLN)
Probabilistic Logic Networks with:
- **Deduction**: A→B, B→C ⊢ A→C
- **Induction**: A→B, A→C ⊢ B→C
- **Abduction**: B→C, A→C ⊢ A→B
- **Revision**: Combine evidence
- **Forward chaining**: Automatic inference
- **Backward chaining**: Goal-directed proving

#### Attention Bank (ECAN)
Economic Attention Network featuring:
- Importance spreading along links
- STI decay over time
- Normalization to maintain total STI
- Attention consolidation (STI → LTI)
- Forgetting low-importance atoms

#### Learning
- **Pattern mining**: Discover frequent patterns
- **Correlation learning**: Find co-occurring atoms
- **Reinforcement learning**: Update importance based on rewards
- **Pattern completion**: Predict missing parts
- **Concept formation**: Cluster similar atoms

## API Reference

### AtomSpace

```lua
local atomspace = opencog.AtomSpace.new()

atomspace:add(atom)                    -- Add atom
atomspace:remove(atom)                 -- Remove atom
atomspace:get_atom(id)                 -- Get by ID
atomspace:get_node(type, name)         -- Get node by type and name
atomspace:get_atoms_by_type(type)      -- Get all atoms of type
atomspace:query(pattern)               -- Pattern matching
atomspace:size()                       -- Get atom count
atomspace:clear()                      -- Clear all atoms
```

### Node

```lua
local node = opencog.Node.new("ConceptNode", "name")

node:is_node()                         -- Returns true
node:is_link()                         -- Returns false
node:get_hash()                        -- Get hash
node:set_truth_value(tv)               -- Set truth value
node:get_truth_value()                 -- Get truth value
node:set_attention_value(av)           -- Set attention value
node:get_attention_value()             -- Get attention value
node:get_incoming_set()                -- Get links containing this node
```

### Link

```lua
local link = opencog.Link.new("InheritanceLink", {node1, node2})

link:is_node()                         -- Returns false
link:is_link()                         -- Returns true
link:get_arity()                       -- Get number of outgoing atoms
link:get_outgoing_set()                -- Get outgoing atoms
```

### Reasoner

```lua
local reasoner = opencog.Reasoner.new(atomspace)

reasoner:deduction(ab, bc)             -- A→B, B→C ⊢ A→C
reasoner:induction(ab, ac)             -- A→B, A→C ⊢ B→C
reasoner:abduction(bc, ac)             -- B→C, A→C ⊢ A→B
reasoner:revision(tv1, tv2)            -- Combine truth values
reasoner:forward_chain(max_steps)      -- Forward chaining
reasoner:backward_chain(goal, depth)   -- Backward chaining
```

### Attention Bank

```lua
local attention = opencog.AttentionBank.new(atomspace, config)

attention:stimulate(atom, amount)      -- Increase STI
attention:update_attention()           -- Update dynamics
attention:get_attentional_focus()      -- Get focused atoms
attention:get_top_atoms(n)             -- Get top N atoms
attention:consolidate_attention()      -- Convert STI to LTI
attention:forget_low_importance(threshold) -- Remove unimportant atoms
```

### Learning

```lua
local learning = opencog.Learning.new(atomspace)

learning:mine_frequent_patterns(support)    -- Find patterns
learning:learn_correlations(threshold)      -- Find correlations
learning:reinforce_atom(atom, reward)       -- Reinforce importance
learning:complete_pattern(partial)          -- Complete pattern
learning:form_concepts(atoms, threshold)    -- Cluster atoms
```

## Running Examples

```bash
cd lua

# Run the basic example
lua5.3 examples/basic_example.lua

# Run tests
lua5.3 tests/test_opencog.lua
```

## Testing

The test suite includes 15 comprehensive tests covering:

- AtomSpace operations
- Node and Link creation
- TruthValue and AttentionValue
- Pattern matching (including wildcards)
- PLN reasoning
- Attention allocation
- Learning and pattern mining
- Incoming sets

All tests pass successfully.

## Performance Considerations

- **In-memory storage**: All atoms stored in Lua tables
- **Indexing**: Fast lookup by ID, type, and type+name
- **Pattern matching**: O(n) where n is atoms of matching type
- **Attention**: O(n) for updates, O(n log n) for sorting

## Future Enhancements

- Persistent storage backend
- Distributed AtomSpace
- More PLN inference rules
- Advanced pattern matching (graph isomorphism)
- Integration with neural networks
- MeTTa language support

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Acknowledgments

Based on the OpenCog framework:
- OpenCog Foundation
- Ben Goertzel (OpenCog Prime, Hyperon)
- The AGI research community

## References

- [OpenCog Wiki](https://wiki.opencog.org/)
- [OpenCog Hyperon](https://github.com/trueagi-io/hyperon-experimental)
- [Probabilistic Logic Networks](https://wiki.opencog.org/w/PLN)
- [ECAN](https://wiki.opencog.org/w/Attention_Allocation)
