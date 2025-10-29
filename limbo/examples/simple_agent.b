implement SimpleAgent;

#
# Simple Cognitive Agent Example
# Demonstrates basic usage of the OpenCog Limbo implementation
#

include "sys.m";
    sys: Sys;
include "draw.m";
include "../cogcore.m";
    cogcore: CogCore;
    CogCore, CogConfig: import cogcore;
include "../perception.m";
    perception: Perception;
    SensoryInput: import perception;
include "../action.m";
    action: Action;
    Action: import action;
include "../atomspace.m";
    atomspace: Atomspace;
    Node, Link, TruthValue: import atomspace;

init(nil: ref Draw->Context, args: list of string)
{
    sys = load Sys Sys->PATH;
    
    # Load modules
    cogcore = load CogCore CogCore->PATH;
    if (cogcore == nil) {
        sys->print("Cannot load CogCore module\n");
        raise "fail:load";
    }
    
    perception = load Perception Perception->PATH;
    if (perception == nil) {
        sys->print("Cannot load Perception module\n");
        raise "fail:load";
    }
    
    action = load Action Action->PATH;
    if (action == nil) {
        sys->print("Cannot load Action module\n");
        raise "fail:load";
    }
    
    atomspace = load Atomspace Atomspace->PATH;
    if (atomspace == nil) {
        sys->print("Cannot load Atomspace module\n");
        raise "fail:load";
    }
    
    sys->print("Simple Cognitive Agent Example\n");
    sys->print("================================\n\n");
    
    # Create configuration
    config := CogConfig.mk();
    config.visual_dim = 784;
    config.audio_dim = 256;
    config.wm_capacity = 7;
    config.learning_rate = 0.01;
    
    sys->print("Creating cognitive core with configuration:\n");
    sys->print("  Visual dimension: %d\n", config.visual_dim);
    sys->print("  Audio dimension: %d\n", config.audio_dim);
    sys->print("  Working memory capacity: %d\n", config.wm_capacity);
    sys->print("  Learning rate: %.3f\n", config.learning_rate);
    sys->print("\n");
    
    # Create cognitive core
    core := CogCore.mk(config);
    
    # Add some basic knowledge to the AtomSpace
    sys->print("Initializing knowledge base...\n");
    
    # Create concept nodes
    cat := core.space.add_node("ConceptNode", "cat");
    animal := core.space.add_node("ConceptNode", "animal");
    mammal := core.space.add_node("ConceptNode", "mammal");
    
    # Create inheritance relationships
    outgoing1 := array[2] of ref atomspace->Atom;
    outgoing1[0] = ref cat.atom;
    outgoing1[1] = ref mammal.atom;
    cat_is_mammal := core.space.add_link("InheritanceLink", outgoing1);
    
    outgoing2 := array[2] of ref atomspace->Atom;
    outgoing2[0] = ref mammal.atom;
    outgoing2[1] = ref animal.atom;
    mammal_is_animal := core.space.add_link("InheritanceLink", outgoing2);
    
    sys->print("  Added knowledge: cat -> mammal -> animal\n");
    sys->print("  AtomSpace size: %d atoms\n\n", core.space.size());
    
    # Add some actions
    sys->print("Adding actions to action module...\n");
    
    action_params: list of real = nil;
    
    move_forward := Action.mk("move_forward", action_params);
    move_forward.expected_reward = 0.5;
    core.action_sel.add_action(move_forward);
    
    turn_left := Action.mk("turn_left", action_params);
    turn_left.expected_reward = 0.3;
    core.action_sel.add_action(turn_left);
    
    turn_right := Action.mk("turn_right", action_params);
    turn_right.expected_reward = 0.3;
    core.action_sel.add_action(turn_right);
    
    grab_object := Action.mk("grab_object", action_params);
    grab_object.expected_reward = 0.8;
    core.action_sel.add_action(grab_object);
    
    sys->print("  Added 4 actions\n\n");
    
    # Run cognitive cycles
    sys->print("Running cognitive cycles...\n");
    sys->print("----------------------------\n\n");
    
    run_agent_simulation(core, config);
    
    sys->print("\nAgent simulation complete!\n");
}

run_agent_simulation(core: ref CogCore->CogCore, config: ref CogCore->CogConfig)
{
    num_cycles := 50;
    
    for (cycle := 0; cycle < num_cycles; cycle++) {
        # Create sensory input (simulated)
        visual := array[config.visual_dim] of real;
        audio := array[config.audio_dim] of real;
        
        # Simulate varying sensory input
        base_val := 0.5;
        for (i := 0; i < len visual; i++)
            visual[i] = base_val;
        for (i := 0; i < len audio; i++)
            audio[i] = 0.1;
        
        input := SensoryInput.mk(visual, audio, nil);
        
        # Simulate reward (varies by cycle)
        reward := 0.5 + real(cycle % 10) / 20.0;
        
        # Run cognitive cycle
        selected_action := core.cognitive_cycle(input, reward);
        
        # Print results
        if (selected_action != nil) {
            sys->print("Cycle %2d: Action=%s, Reward=%.2f, ExpectedReward=%.2f\n",
                      cycle, selected_action.name, reward, 
                      selected_action.expected_reward);
        } else {
            sys->print("Cycle %2d: No action selected, Reward=%.2f\n",
                      cycle, reward);
        }
        
        # Print detailed stats every 10 cycles
        if (cycle > 0 && cycle % 10 == 0) {
            sys->print("\n");
            stats := core.get_stats();
            sys->print("%s\n", stats);
        }
    }
    
    sys->print("\n");
    sys->print("Final Statistics:\n");
    sys->print("=================\n");
    stats := core.get_stats();
    sys->print("%s\n", stats);
}
