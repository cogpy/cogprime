implement Reasoning;

#
# Reasoning Module - Pattern recognition and inference for OpenCog
#

include "sys.m";
    sys: Sys;
include "atomspace.m";
    atomspace: Atomspace;
    AtomSpace, Node, Link, Atom, TruthValue: import atomspace;

# Working memory item
WorkingMemoryItem: adt {
    atom:       ref Atom;        # The atom in working memory
    activation: real;            # Activation level
    timestamp:  int;             # When it entered working memory
    
    mk:         fn(atom: ref Atom, activation: real): ref WorkingMemoryItem;
};

# Episodic memory entry
EpisodicMemory: adt {
    event:      ref Link;        # Event representation
    timestamp:  int;             # When the event occurred
    importance: real;            # Event importance
    
    mk:         fn(event: ref Link, importance: real): ref EpisodicMemory;
};

# Reasoning module
ReasoningModule: adt {
    space:          ref AtomSpace;  # Reference to AtomSpace
    working_memory: list of ref WorkingMemoryItem; # Working memory
    episodic_mem:   list of ref EpisodicMemory; # Episodic memory
    wm_capacity:    int;            # Working memory capacity
    attention_decay: real;          # Attention decay rate
    
    mk:             fn(space: ref AtomSpace, wm_capacity: int, 
                       attention_decay: real): ref ReasoningModule;
    infer:          fn(rm: self ref ReasoningModule, 
                       premises: list of ref Atom): list of ref Link;
    recognize_pattern: fn(rm: self ref ReasoningModule, 
                         pattern: ref Atom): list of ref Atom;
    update_working_memory: fn(rm: self ref ReasoningModule, 
                             atoms: list of ref Atom): int;
    store_episode: fn(rm: self ref ReasoningModule, 
                     event: ref Link, importance: real): int;
    retrieve_episodes: fn(rm: self ref ReasoningModule, 
                         query: ref Atom): list of ref EpisodicMemory;
};

init(nil: ref Draw->Context, args: list of string)
{
    sys = load Sys Sys->PATH;
    atomspace = load Atomspace Atomspace->PATH;
    if (atomspace == nil) {
        sys->print("Cannot load Atomspace module\n");
        raise "fail:load";
    }
}

# Implementation of WorkingMemoryItem
WorkingMemoryItem.mk(atom: ref Atom, activation: real): ref WorkingMemoryItem
{
    item := ref WorkingMemoryItem;
    item.atom = atom;
    item.activation = activation;
    item.timestamp = sys->millisec();
    return item;
}

# Implementation of EpisodicMemory
EpisodicMemory.mk(event: ref Link, importance: real): ref EpisodicMemory
{
    mem := ref EpisodicMemory;
    mem.event = event;
    mem.importance = importance;
    mem.timestamp = sys->millisec();
    return mem;
}

# Implementation of ReasoningModule
ReasoningModule.mk(space: ref AtomSpace, wm_capacity: int, 
                   attention_decay: real): ref ReasoningModule
{
    rm := ref ReasoningModule;
    rm.space = space;
    rm.working_memory = nil;
    rm.episodic_mem = nil;
    rm.wm_capacity = wm_capacity;
    rm.attention_decay = attention_decay;
    return rm;
}

ReasoningModule.infer(rm: self ref ReasoningModule, 
                     premises: list of ref Atom): list of ref Link
{
    conclusions: list of ref Link = nil;
    
    # Simple inference: for each premise, find related atoms
    for (p := premises; p != nil; p = tl p) {
        premise := hd p;
        
        # Pattern match against atoms in the space
        # This is a simplified inference - real implementation would use
        # probabilistic logic networks, forward/backward chaining, etc.
        
        # For inheritance relationships, do transitive inference
        if (premise.atomtype == "InheritanceLink") {
            # Would extract A and B from InheritanceLink(A, B)
            # and find InheritanceLink(B, C) to infer InheritanceLink(A, C)
        }
    }
    
    return conclusions;
}

ReasoningModule.recognize_pattern(rm: self ref ReasoningModule, 
                                  pattern: ref Atom): list of ref Atom
{
    # Use AtomSpace query to find matching patterns
    matches := rm.space.query(pattern);
    return matches;
}

ReasoningModule.update_working_memory(rm: self ref ReasoningModule, 
                                     atoms: list of ref Atom): int
{
    # Add new atoms to working memory
    for (a := atoms; a != nil; a = tl a) {
        atom := hd a;
        
        # Calculate activation based on attention value
        activation := 1.0;
        if (atom.av != nil)
            activation = atom.av.sti;
        
        # Add to working memory
        item := WorkingMemoryItem.mk(atom, activation);
        rm.working_memory = item :: rm.working_memory;
    }
    
    # Enforce capacity limit - remove lowest activation items
    # Count items
    count := 0;
    for (wm := rm.working_memory; wm != nil; wm = tl wm)
        count++;
    
    # If over capacity, remove excess items
    if (count > rm.wm_capacity) {
        # Sort by activation and keep top items
        # Simplified: just keep first wm_capacity items
        new_wm: list of ref WorkingMemoryItem = nil;
        i := 0;
        for (wm := rm.working_memory; wm != nil && i < rm.wm_capacity; wm = tl wm) {
            new_wm = hd wm :: new_wm;
            i++;
        }
        rm.working_memory = new_wm;
    }
    
    # Apply decay to all items
    for (wm := rm.working_memory; wm != nil; wm = tl wm) {
        item := hd wm;
        item.activation = item.activation * rm.attention_decay;
    }
    
    return 1;
}

ReasoningModule.store_episode(rm: self ref ReasoningModule, 
                              event: ref Link, importance: real): int
{
    # Store an episodic memory
    episode := EpisodicMemory.mk(event, importance);
    rm.episodic_mem = episode :: rm.episodic_mem;
    return 1;
}

ReasoningModule.retrieve_episodes(rm: self ref ReasoningModule, 
                                  query: ref Atom): list of ref EpisodicMemory
{
    results: list of ref EpisodicMemory = nil;
    
    # Simple retrieval - match by atom type
    for (em := rm.episodic_mem; em != nil; em = tl em) {
        episode := hd em;
        if (episode.event != nil && episode.event.atom.atomtype == query.atomtype)
            results = episode :: results;
    }
    
    return results;
}
