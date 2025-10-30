implement Learning;

#
# Learning Module - Experience-based adaptation for OpenCog
#

include "sys.m";
    sys: Sys;
include "atomspace.m";
    atomspace: Atomspace;
    AtomSpace, Node, Link, TruthValue: import atomspace;

# Experience record
Experience: adt {
    state:      list of ref Node;   # State representation
    action:     string;              # Action taken
    reward:     real;                # Reward received
    next_state: list of ref Node;   # Resulting state
    timestamp:  int;                 # When experience occurred
    
    mk:         fn(state: list of ref Node, action: string, 
                   reward: real, next_state: list of ref Node): ref Experience;
};

# Skill representation
Skill: adt {
    name:       string;              # Skill name
    pattern:    ref Link;            # Pattern that triggers skill
    proficiency: real;               # Skill proficiency level
    uses:       int;                 # Number of times used
    
    mk:         fn(name: string, pattern: ref Link): ref Skill;
    practice:   fn(s: self ref Skill, success: int): int;
};

# Learning module
LearningModule: adt {
    space:          ref AtomSpace;   # Reference to AtomSpace
    experiences:    list of ref Experience; # Experience buffer
    skills:         list of ref Skill;  # Learned skills
    learning_rate:  real;            # Learning rate
    buffer_size:    int;             # Experience buffer size
    
    mk:             fn(space: ref AtomSpace, learning_rate: real, 
                       buffer_size: int): ref LearningModule;
    learn_from_experience: fn(lm: self ref LearningModule, 
                             exp: ref Experience): int;
    update_skill:   fn(lm: self ref LearningModule, 
                       skill: ref Skill, success: int): int;
    get_skill:      fn(lm: self ref LearningModule, 
                       pattern: ref Link): ref Skill;
    consolidate:    fn(lm: self ref LearningModule): int;
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

# Implementation of Experience
Experience.mk(state: list of ref Node, action: string, 
              reward: real, next_state: list of ref Node): ref Experience
{
    exp := ref Experience;
    exp.state = state;
    exp.action = action;
    exp.reward = reward;
    exp.next_state = next_state;
    exp.timestamp = sys->millisec();
    return exp;
}

# Implementation of Skill
Skill.mk(name: string, pattern: ref Link): ref Skill
{
    skill := ref Skill;
    skill.name = name;
    skill.pattern = pattern;
    skill.proficiency = 0.0;
    skill.uses = 0;
    return skill;
}

Skill.practice(s: self ref Skill, success: int): int
{
    # Update proficiency based on practice
    s.uses++;
    
    # Simple learning curve - proficiency increases with practice
    if (success) {
        # Increase proficiency
        s.proficiency = s.proficiency + (1.0 - s.proficiency) * 0.1;
    } else {
        # Slight decrease on failure
        s.proficiency = s.proficiency * 0.95;
    }
    
    # Clamp to [0, 1]
    if (s.proficiency < 0.0)
        s.proficiency = 0.0;
    if (s.proficiency > 1.0)
        s.proficiency = 1.0;
    
    return 1;
}

# Implementation of LearningModule
LearningModule.mk(space: ref AtomSpace, learning_rate: real, 
                  buffer_size: int): ref LearningModule
{
    lm := ref LearningModule;
    lm.space = space;
    lm.experiences = nil;
    lm.skills = nil;
    lm.learning_rate = learning_rate;
    lm.buffer_size = buffer_size;
    return lm;
}

LearningModule.learn_from_experience(lm: self ref LearningModule, 
                                     exp: ref Experience): int
{
    # Add experience to buffer
    lm.experiences = exp :: lm.experiences;
    
    # Maintain buffer size
    count := 0;
    for (e := lm.experiences; e != nil; e = tl e)
        count++;
    
    if (count > lm.buffer_size) {
        # Remove oldest experiences
        new_exp: list of ref Experience = nil;
        i := 0;
        for (e := lm.experiences; e != nil && i < lm.buffer_size; e = tl e) {
            new_exp = hd e :: new_exp;
            i++;
        }
        lm.experiences = new_exp;
    }
    
    # Update truth values in AtomSpace based on experience
    # This is where reinforcement learning would occur
    
    # For each state node, update its truth value based on reward
    for (s := exp.state; s != nil; s = tl s) {
        node := hd s;
        if (node != nil && node.atom.tv != nil) {
            # Update strength based on reward
            old_strength := node.atom.tv.strength;
            new_strength := old_strength + lm.learning_rate * (exp.reward - old_strength);
            
            # Clamp to [0, 1]
            if (new_strength < 0.0)
                new_strength = 0.0;
            if (new_strength > 1.0)
                new_strength = 1.0;
            
            node.atom.tv.strength = new_strength;
            
            # Increase confidence
            old_conf := node.atom.tv.confidence;
            new_conf := old_conf + (1.0 - old_conf) * 0.01;
            node.atom.tv.confidence = new_conf;
        }
    }
    
    return 1;
}

LearningModule.update_skill(lm: self ref LearningModule, 
                           skill: ref Skill, success: int): int
{
    # Update skill proficiency
    skill.practice(success);
    return 1;
}

LearningModule.get_skill(lm: self ref LearningModule, 
                        pattern: ref Link): ref Skill
{
    # Find skill matching pattern
    for (s := lm.skills; s != nil; s = tl s) {
        skill := hd s;
        if (skill.pattern != nil && pattern != nil) {
            # Simple match by atom type
            if (skill.pattern.atom.atomtype == pattern.atom.atomtype)
                return skill;
        }
    }
    return nil;
}

LearningModule.consolidate(lm: self ref LearningModule): int
{
    # Consolidate experiences into long-term knowledge
    # This is where learned patterns would be extracted and
    # stored as new links in the AtomSpace
    
    # Count experiences
    count := 0;
    for (e := lm.experiences; e != nil; e = tl e)
        count++;
    
    if (count < 10) {
        # Not enough experiences to consolidate
        return 0;
    }
    
    # Extract common patterns
    # This is simplified - real implementation would use
    # statistical pattern mining
    
    # For now, just create a skill from the most common state-action pair
    # Would analyze lm.experiences to find patterns
    
    return 1;
}
