Action: module
{
    PATH: con "/dis/cogprime/action.dis";
    
    Action: adt {
        name:       string;
        params:     list of real;
        expected_reward: real;
        
        mk:         fn(name: string, params: list of real): ref Action;
        execute:    fn(a: self ref Action): int;
    };
    
    Goal: adt {
        node:       ref Atomspace->Node;
        priority:   real;
        satisfied:  int;
        
        mk:         fn(node: ref Atomspace->Node, priority: real): ref Goal;
    };
    
    ActionModule: adt {
        space:          ref Atomspace->AtomSpace;
        action_space:   list of ref Action;
        current_goal:   ref Goal;
        exploration_rate: real;
        
        mk:             fn(space: ref Atomspace->AtomSpace, 
                           exploration_rate: real): ref ActionModule;
        select_action:  fn(am: self ref ActionModule, 
                           state: list of ref Atomspace->Node): ref Action;
        add_action:     fn(am: self ref ActionModule, 
                           action: ref Action): int;
        set_goal:       fn(am: self ref ActionModule, 
                           goal: ref Goal): int;
        evaluate:       fn(am: self ref ActionModule, 
                           action: ref Action, reward: real): int;
    };
    
    init: fn(nil: ref Draw->Context, args: list of string);
};
