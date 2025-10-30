Reasoning: module
{
    PATH: con "/dis/cogprime/reasoning.dis";
    
    WorkingMemoryItem: adt {
        atom:       ref Atomspace->Atom;
        activation: real;
        timestamp:  int;
        
        mk:         fn(atom: ref Atomspace->Atom, activation: real): ref WorkingMemoryItem;
    };
    
    EpisodicMemory: adt {
        event:      ref Atomspace->Link;
        timestamp:  int;
        importance: real;
        
        mk:         fn(event: ref Atomspace->Link, importance: real): ref EpisodicMemory;
    };
    
    ReasoningModule: adt {
        space:          ref Atomspace->AtomSpace;
        working_memory: list of ref WorkingMemoryItem;
        episodic_mem:   list of ref EpisodicMemory;
        wm_capacity:    int;
        attention_decay: real;
        
        mk:             fn(space: ref Atomspace->AtomSpace, 
                           wm_capacity: int, attention_decay: real): ref ReasoningModule;
        infer:          fn(rm: self ref ReasoningModule, 
                           premises: list of ref Atomspace->Atom): list of ref Atomspace->Link;
        recognize_pattern: fn(rm: self ref ReasoningModule, 
                             pattern: ref Atomspace->Atom): list of ref Atomspace->Atom;
        update_working_memory: fn(rm: self ref ReasoningModule, 
                                 atoms: list of ref Atomspace->Atom): int;
        store_episode: fn(rm: self ref ReasoningModule, 
                         event: ref Atomspace->Link, importance: real): int;
        retrieve_episodes: fn(rm: self ref ReasoningModule, 
                             query: ref Atomspace->Atom): list of ref EpisodicMemory;
    };
    
    init: fn(nil: ref Draw->Context, args: list of string);
};
