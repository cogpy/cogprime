-- OpenCog in Pure Lua
-- Main module initialization

local opencog = {
    -- Version
    VERSION = "1.0.0",
    
    -- Core components
    Atom = require('opencog.atom').Atom,
    TruthValue = require('opencog.atom').TruthValue,
    AttentionValue = require('opencog.atom').AttentionValue,
    Wildcard = require('opencog.atom').Wildcard,
    Node = require('opencog.node'),
    Link = require('opencog.link'),
    AtomSpace = require('opencog.atomspace'),
    
    -- Advanced components
    PatternMatcher = require('opencog.pattern_matcher'),
    AttentionBank = require('opencog.attention'),
    Reasoner = require('opencog.reasoner'),
    Learning = require('opencog.learning'),
}

-- Convenience function to create a node
function opencog.create_node(atom_type, name)
    return opencog.Node.new(atom_type, name)
end

-- Convenience function to create a link
function opencog.create_link(atom_type, outgoing_set)
    return opencog.Link.new(atom_type, outgoing_set)
end

-- Convenience function to create an atomspace
function opencog.create_atomspace()
    return opencog.AtomSpace.new()
end

-- Print OpenCog information
function opencog.info()
    print("OpenCog in Pure Lua v" .. opencog.VERSION)
    print("An implementation of the OpenCog cognitive architecture")
    print("")
    print("Components available:")
    print("  - AtomSpace: Hypergraph knowledge representation")
    print("  - Node & Link: Atoms for building knowledge graphs")
    print("  - TruthValue: Probabilistic truth values")
    print("  - AttentionValue: Economic attention allocation")
    print("  - PatternMatcher: Advanced pattern matching")
    print("  - AttentionBank: ECAN attention allocation")
    print("  - Reasoner: PLN probabilistic logic reasoning")
    print("  - Learning: Pattern mining and reinforcement learning")
end

return opencog
