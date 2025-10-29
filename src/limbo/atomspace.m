Atomspace: module
{
    PATH: con "/dis/cogprime/atomspace.dis";
    
    # Atom types
    Atom: adt {
        id:         string;
        atomtype:   string;
        tv:         ref TruthValue;
        av:         ref AttentionValue;
        values:     list of (string, string);
        
        is_node:    fn(a: self ref Atom): int;
        is_link:    fn(a: self ref Atom): int;
        get_hash:   fn(a: self ref Atom): int;
        to_string:  fn(a: self ref Atom): string;
    };
    
    Node: adt {
        atom:       Atom;
        name:       string;
        
        mk:         fn(atomtype: string, name: string): ref Node;
        is_node:    fn(n: self ref Node): int;
        is_link:    fn(n: self ref Node): int;
        get_hash:   fn(n: self ref Node): int;
        to_string:  fn(n: self ref Node): string;
    };
    
    Link: adt {
        atom:       Atom;
        outgoing:   array of ref Atom;
        
        mk:         fn(atomtype: string, outgoing: array of ref Atom): ref Link;
        is_node:    fn(l: self ref Link): int;
        is_link:    fn(l: self ref Link): int;
        get_hash:   fn(l: self ref Link): int;
        get_arity:  fn(l: self ref Link): int;
        to_string:  fn(l: self ref Link): string;
    };
    
    TruthValue: adt {
        strength:   real;
        confidence: real;
        
        mk:         fn(strength: real, confidence: real): ref TruthValue;
        to_string:  fn(tv: self ref TruthValue): string;
    };
    
    AttentionValue: adt {
        sti:        real;
        lti:        real;
        vlti:       int;
        
        mk:         fn(sti: real, lti: real, vlti: int): ref AttentionValue;
        to_string:  fn(av: self ref AttentionValue): string;
    };
    
    AtomSpace: adt {
        atoms_by_id:        list of (string, ref Atom);
        nodes_by_type_name: list of (string, ref Node);
        atoms_by_type:      list of (string, list of ref Atom);
        
        mk:         fn(): ref AtomSpace;
        add_node:   fn(space: self ref AtomSpace, atomtype: string, name: string): ref Node;
        add_link:   fn(space: self ref AtomSpace, atomtype: string, outgoing: array of ref Atom): ref Link;
        add:        fn(space: self ref AtomSpace, atom: ref Atom): ref Atom;
        remove:     fn(space: self ref AtomSpace, atom: ref Atom): int;
        get_atom:   fn(space: self ref AtomSpace, id: string): ref Atom;
        get_node:   fn(space: self ref AtomSpace, atomtype: string, name: string): ref Node;
        get_atoms_by_type: fn(space: self ref AtomSpace, atomtype: string): list of ref Atom;
        query:      fn(space: self ref AtomSpace, pattern: ref Atom): list of ref Atom;
        size:       fn(space: self ref AtomSpace): int;
    };
    
    init: fn(nil: ref Draw->Context, args: list of string);
};
