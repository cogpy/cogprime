Perception: module
{
    PATH: con "/dis/cogprime/perception.dis";
    
    SensoryInput: adt {
        visual:     array of real;
        auditory:   array of real;
        tactile:    array of real;
        timestamp:  int;
        
        mk:         fn(visual: array of real, auditory: array of real, 
                       tactile: array of real): ref SensoryInput;
    };
    
    AttentionMechanism: adt {
        threshold:  real;
        decay:      real;
        
        mk:         fn(threshold: real, decay: real): ref AttentionMechanism;
        allocate:   fn(am: self ref AttentionMechanism, 
                       space: ref Atomspace->AtomSpace, 
                       inputs: array of ref SensoryInput): int;
        spread:     fn(am: self ref AttentionMechanism, 
                       space: ref Atomspace->AtomSpace): int;
    };
    
    PerceptionModule: adt {
        space:      ref Atomspace->AtomSpace;
        attention:  ref AttentionMechanism;
        visual_dim: int;
        audio_dim:  int;
        
        mk:         fn(space: ref Atomspace->AtomSpace, 
                       visual_dim: int, audio_dim: int): ref PerceptionModule;
        process:    fn(pm: self ref PerceptionModule, 
                       input: ref SensoryInput): list of ref Atomspace->Node;
        encode:     fn(pm: self ref PerceptionModule,
                       data: array of real, modality: string): ref Atomspace->Node;
    };
    
    init: fn(nil: ref Draw->Context, args: list of string);
};
