Learning: module
{
    PATH: con "/dis/cogprime/learning.dis";
    
    Experience: adt {
        state:      list of ref Atomspace->Node;
        action:     string;
        reward:     real;
        next_state: list of ref Atomspace->Node;
        timestamp:  int;
        
        mk:         fn(state: list of ref Atomspace->Node, action: string, 
                       reward: real, next_state: list of ref Atomspace->Node): ref Experience;
    };
    
    Skill: adt {
        name:       string;
        pattern:    ref Atomspace->Link;
        proficiency: real;
        uses:       int;
        
        mk:         fn(name: string, pattern: ref Atomspace->Link): ref Skill;
        practice:   fn(s: self ref Skill, success: int): int;
    };
    
    LearningModule: adt {
        space:          ref Atomspace->AtomSpace;
        experiences:    list of ref Experience;
        skills:         list of ref Skill;
        learning_rate:  real;
        buffer_size:    int;
        
        mk:             fn(space: ref Atomspace->AtomSpace, 
                           learning_rate: real, buffer_size: int): ref LearningModule;
        learn_from_experience: fn(lm: self ref LearningModule, 
                                 exp: ref Experience): int;
        update_skill:   fn(lm: self ref LearningModule, 
                           skill: ref Skill, success: int): int;
        get_skill:      fn(lm: self ref LearningModule, 
                           pattern: ref Atomspace->Link): ref Skill;
        consolidate:    fn(lm: self ref LearningModule): int;
    };
    
    init: fn(nil: ref Draw->Context, args: list of string);
};
