implement CogCore;

#
# CogCore - Integrated cognitive core for OpenCog
# Combines all cognitive modules into a unified system
#

include "sys.m";
    sys: Sys;
include "atomspace.m";
    atomspace: Atomspace;
    AtomSpace: import atomspace;
include "perception.m";
    perception: Perception;
    PerceptionModule, SensoryInput: import perception;
include "reasoning.m";
    reasoning: Reasoning;
    ReasoningModule: import reasoning;
include "action.m";
    action: Action;
    ActionModule, Action: import action;
include "learning.m";
    learning: Learning;
    LearningModule, Experience: import learning;

# Cognitive core configuration
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

# Main cognitive core
CogCore: adt {
    config:         ref CogConfig;
    space:          ref AtomSpace;
    perception:     ref PerceptionModule;
    reasoning:      ref ReasoningModule;
    action_sel:     ref ActionModule;
    learning:       ref LearningModule;
    cycle_count:    int;
    
    mk:             fn(config: ref CogConfig): ref CogCore;
    cognitive_cycle: fn(core: self ref CogCore, 
                       input: ref SensoryInput, reward: real): ref Action;
    get_stats:      fn(core: self ref CogCore): string;
};

init(nil: ref Draw->Context, args: list of string)
{
    sys = load Sys Sys->PATH;
    
    atomspace = load Atomspace Atomspace->PATH;
    if (atomspace == nil) {
        sys->print("Cannot load Atomspace module\n");
        raise "fail:load";
    }
    
    perception = load Perception Perception->PATH;
    if (perception == nil) {
        sys->print("Cannot load Perception module\n");
        raise "fail:load";
    }
    
    reasoning = load Reasoning Reasoning->PATH;
    if (reasoning == nil) {
        sys->print("Cannot load Reasoning module\n");
        raise "fail:load";
    }
    
    action = load Action Action->PATH;
    if (action == nil) {
        sys->print("Cannot load Action module\n");
        raise "fail:load";
    }
    
    learning = load Learning Learning->PATH;
    if (learning == nil) {
        sys->print("Cannot load Learning module\n");
        raise "fail:load";
    }
}

# Implementation of CogConfig
CogConfig.mk(): ref CogConfig
{
    config := ref CogConfig;
    config.visual_dim = 784;
    config.audio_dim = 256;
    config.wm_capacity = 7;
    config.attention_decay = 0.95;
    config.exploration_rate = 0.1;
    config.learning_rate = 0.001;
    config.buffer_size = 1000;
    return config;
}

CogConfig.default(): ref CogConfig
{
    return CogConfig.mk();
}

# Implementation of CogCore
CogCore.mk(config: ref CogConfig): ref CogCore
{
    core := ref CogCore;
    
    # Use default config if none provided
    if (config == nil)
        config = CogConfig.default();
    
    core.config = config;
    
    # Initialize AtomSpace
    core.space = AtomSpace.mk();
    
    # Initialize cognitive modules
    core.perception = PerceptionModule.mk(
        core.space,
        config.visual_dim,
        config.audio_dim
    );
    
    core.reasoning = ReasoningModule.mk(
        core.space,
        config.wm_capacity,
        config.attention_decay
    );
    
    core.action_sel = ActionModule.mk(
        core.space,
        config.exploration_rate
    );
    
    core.learning = LearningModule.mk(
        core.space,
        config.learning_rate,
        config.buffer_size
    );
    
    core.cycle_count = 0;
    
    return core;
}

CogCore.cognitive_cycle(core: self ref CogCore, 
                       input: ref SensoryInput, reward: real): ref Action
{
    # Increment cycle counter
    core.cycle_count++;
    
    # 1. Perception - Process sensory input
    percepts := core.perception.process(input);
    
    # 2. Reasoning - Update working memory and recognize patterns
    core.reasoning.update_working_memory(percepts);
    
    # Get current working memory state
    wm_atoms: list of ref atomspace->Node = nil;
    for (wm := core.reasoning.working_memory; wm != nil; wm = tl wm) {
        item := hd wm;
        # Would need to convert Atom to Node - simplified here
        # wm_atoms = item.atom :: wm_atoms;
    }
    
    # 3. Action Selection - Choose action based on state
    selected_action := core.action_sel.select_action(wm_atoms);
    
    # 4. Learning - Learn from previous experience
    if (selected_action != nil && percepts != nil) {
        # Create experience record
        exp := Experience.mk(percepts, selected_action.name, reward, nil);
        core.learning.learn_from_experience(exp);
        
        # Update action evaluation
        core.action_sel.evaluate(selected_action, reward);
    }
    
    # 5. Attention allocation
    inputs := array[1] of ref SensoryInput;
    inputs[0] = input;
    core.perception.attention.allocate(core.space, inputs);
    core.perception.attention.spread(core.space);
    
    # 6. Consolidation (periodic)
    if (core.cycle_count % 100 == 0) {
        core.learning.consolidate();
    }
    
    return selected_action;
}

CogCore.get_stats(core: self ref CogCore): string
{
    stats := sys->sprint("CogCore Statistics:\n");
    stats += sys->sprint("  Cycles: %d\n", core.cycle_count);
    stats += sys->sprint("  AtomSpace size: %d\n", core.space.size());
    
    # Count working memory items
    wm_count := 0;
    for (wm := core.reasoning.working_memory; wm != nil; wm = tl wm)
        wm_count++;
    stats += sys->sprint("  Working memory: %d/%d\n", 
                        wm_count, core.config.wm_capacity);
    
    # Count experiences
    exp_count := 0;
    for (exp := core.learning.experiences; exp != nil; exp = tl exp)
        exp_count++;
    stats += sys->sprint("  Experiences: %d\n", exp_count);
    
    # Count skills
    skill_count := 0;
    for (skill := core.learning.skills; skill != nil; skill = tl skill)
        skill_count++;
    stats += sys->sprint("  Skills: %d\n", skill_count);
    
    return stats;
}
