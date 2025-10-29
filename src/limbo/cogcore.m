CogCore: module
{
    PATH: con "/dis/cogprime/cogcore.dis";
    
    CogConfig: adt {
        visual_dim:     int;
        audio_dim:      int;
        wm_capacity:    int;
        attention_decay: real;
        exploration_rate: real;
        learning_rate:  real;
        buffer_size:    int;
        
        mk:             fn(): ref CogConfig;
        default:        fn(): ref CogConfig;
    };
    
    CogCore: adt {
        config:         ref CogConfig;
        space:          ref Atomspace->AtomSpace;
        perception:     ref Perception->PerceptionModule;
        reasoning:      ref Reasoning->ReasoningModule;
        action_sel:     ref Action->ActionModule;
        learning:       ref Learning->LearningModule;
        cycle_count:    int;
        
        mk:             fn(config: ref CogConfig): ref CogCore;
        cognitive_cycle: fn(core: self ref CogCore, 
                           input: ref Perception->SensoryInput, 
                           reward: real): ref Action->Action;
        get_stats:      fn(core: self ref CogCore): string;
    };
    
    init: fn(nil: ref Draw->Context, args: list of string);
};
