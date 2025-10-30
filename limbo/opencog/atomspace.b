implement Atomspace;

#
# AtomSpace - Core hypergraph database for OpenCog
# Implements a distributed knowledge representation system
#

include "sys.m";
    sys: Sys;

include "draw.m";

# Atom types - base interface for all atoms
Atom: adt {
    id:         string;          # Unique identifier
    atomtype:   string;          # Type of atom (e.g., "ConceptNode", "InheritanceLink")
    tv:         ref TruthValue;  # Truth value
    av:         ref AttentionValue; # Attention value
    values:     list of (string, string); # Additional key-value pairs
    
    # Methods
    is_node:    fn(a: self ref Atom): int;
    is_link:    fn(a: self ref Atom): int;
    get_hash:   fn(a: self ref Atom): int;
    to_string:  fn(a: self ref Atom): string;
};

# Node - represents a vertex in the hypergraph
Node: adt {
    atom:       Atom;            # Base atom data
    name:       string;          # Node name
    
    # Constructor
    mk:         fn(atomtype: string, name: string): ref Node;
    
    # Atom interface implementation
    is_node:    fn(n: self ref Node): int;
    is_link:    fn(n: self ref Node): int;
    get_hash:   fn(n: self ref Node): int;
    to_string:  fn(n: self ref Node): string;
};

# Link - represents a hyperedge in the hypergraph
Link: adt {
    atom:       Atom;            # Base atom data
    outgoing:   array of ref Atom; # Atoms that this link connects
    
    # Constructor
    mk:         fn(atomtype: string, outgoing: array of ref Atom): ref Link;
    
    # Atom interface implementation
    is_node:    fn(l: self ref Link): int;
    is_link:    fn(l: self ref Link): int;
    get_hash:   fn(l: self ref Link): int;
    get_arity:  fn(l: self ref Link): int;
    to_string:  fn(l: self ref Link): string;
};

# TruthValue - represents truth value with strength and confidence
TruthValue: adt {
    strength:   real;            # Truth value strength (0.0 to 1.0)
    confidence: real;            # Confidence in truth value (0.0 to 1.0)
    
    mk:         fn(strength: real, confidence: real): ref TruthValue;
    to_string:  fn(tv: self ref TruthValue): string;
};

# AttentionValue - represents attention allocation
AttentionValue: adt {
    sti:        real;            # Short-Term Importance
    lti:        real;            # Long-Term Importance  
    vlti:       int;             # Very Long-Term Importance flag
    
    mk:         fn(sti: real, lti: real, vlti: int): ref AttentionValue;
    to_string:  fn(av: self ref AttentionValue): string;
};

# AtomSpace - main hypergraph database
AtomSpace: adt {
    atoms_by_id:        list of (string, ref Atom);     # id -> atom mapping
    nodes_by_type_name: list of (string, ref Node);     # (type, name) -> node
    atoms_by_type:      list of (string, list of ref Atom); # type -> atoms
    
    # Constructor
    mk:         fn(): ref AtomSpace;
    
    # Core operations
    add_node:   fn(space: self ref AtomSpace, atomtype: string, name: string): ref Node;
    add_link:   fn(space: self ref AtomSpace, atomtype: string, outgoing: array of ref Atom): ref Link;
    add:        fn(space: self ref AtomSpace, atom: ref Atom): ref Atom;
    remove:     fn(space: self ref AtomSpace, atom: ref Atom): int;
    
    # Retrieval operations
    get_atom:   fn(space: self ref AtomSpace, id: string): ref Atom;
    get_node:   fn(space: self ref AtomSpace, atomtype: string, name: string): ref Node;
    get_atoms_by_type: fn(space: self ref AtomSpace, atomtype: string): list of ref Atom;
    
    # Query operations
    query:      fn(space: self ref AtomSpace, pattern: ref Atom): list of ref Atom;
    
    # Statistics
    size:       fn(space: self ref AtomSpace): int;
};

# Initialize the module
init(nil: ref Draw->Context, args: list of string)
{
    sys = load Sys Sys->PATH;
}

# Implementation of TruthValue
TruthValue.mk(strength: real, confidence: real): ref TruthValue
{
    tv := ref TruthValue;
    
    # Clamp values to [0.0, 1.0]
    if (strength < 0.0)
        tv.strength = 0.0;
    else if (strength > 1.0)
        tv.strength = 1.0;
    else
        tv.strength = strength;
    
    if (confidence < 0.0)
        tv.confidence = 0.0;
    else if (confidence > 1.0)
        tv.confidence = 1.0;
    else
        tv.confidence = confidence;
    
    return tv;
}

TruthValue.to_string(tv: self ref TruthValue): string
{
    return sys->sprint("TruthValue(strength=%.3f, confidence=%.3f)", 
                       tv.strength, tv.confidence);
}

# Implementation of AttentionValue
AttentionValue.mk(sti: real, lti: real, vlti: int): ref AttentionValue
{
    av := ref AttentionValue;
    av.sti = sti;
    av.lti = lti;
    av.vlti = vlti;
    return av;
}

AttentionValue.to_string(av: self ref AttentionValue): string
{
    vlti_str := "false";
    if (av.vlti != 0)
        vlti_str = "true";
    
    return sys->sprint("AttentionValue(sti=%.3f, lti=%.3f, vlti=%s)", 
                       av.sti, av.lti, vlti_str);
}

# Generate unique ID
gen_uuid(): string
{
    # Simple UUID generation - in production would use better method
    return sys->sprint("%d-%d", sys->millisec(), sys->pctl(0, nil));
}

# Implementation of Node
Node.mk(atomtype: string, name: string): ref Node
{
    node := ref Node;
    node.name = name;
    node.atom.id = gen_uuid();
    node.atom.atomtype = atomtype;
    node.atom.tv = TruthValue.mk(1.0, 1.0);
    node.atom.av = AttentionValue.mk(0.0, 0.0, 0);
    node.atom.values = nil;
    return node;
}

Node.is_node(n: self ref Node): int
{
    return 1;
}

Node.is_link(n: self ref Node): int
{
    return 0;
}

Node.get_hash(n: self ref Node): int
{
    # Simple hash function - combine type and name
    hash := 0;
    for (i := 0; i < len n.atom.atomtype; i++)
        hash = hash * 31 + n.atom.atomtype[i];
    for (i := 0; i < len n.name; i++)
        hash = hash * 31 + n.name[i];
    return hash;
}

Node.to_string(n: self ref Node): string
{
    return sys->sprint("%s('%s')", n.atom.atomtype, n.name);
}

# Implementation of Link
Link.mk(atomtype: string, outgoing: array of ref Atom): ref Link
{
    link := ref Link;
    link.atom.id = gen_uuid();
    link.atom.atomtype = atomtype;
    link.atom.tv = TruthValue.mk(1.0, 1.0);
    link.atom.av = AttentionValue.mk(0.0, 0.0, 0);
    link.atom.values = nil;
    link.outgoing = outgoing;
    return link;
}

Link.is_node(l: self ref Link): int
{
    return 0;
}

Link.is_link(l: self ref Link): int
{
    return 1;
}

Link.get_hash(l: self ref Link): int
{
    # Hash based on type and outgoing atoms
    hash := 0;
    for (i := 0; i < len l.atom.atomtype; i++)
        hash = hash * 31 + l.atom.atomtype[i];
    
    for (i := 0; i < len l.outgoing; i++) {
        if (l.outgoing[i] != nil) {
            # Use atom ID for hash
            for (j := 0; j < len l.outgoing[i].id; j++)
                hash = hash * 31 + l.outgoing[i].id[j];
        }
    }
    return hash;
}

Link.get_arity(l: self ref Link): int
{
    return len l.outgoing;
}

Link.to_string(l: self ref Link): string
{
    out := "";
    for (i := 0; i < len l.outgoing; i++) {
        if (i > 0)
            out += ", ";
        # This would need proper atom to_string dispatch
        out += l.outgoing[i].id;
    }
    return sys->sprint("%s(%s)", l.atom.atomtype, out);
}

# Implementation of AtomSpace
AtomSpace.mk(): ref AtomSpace
{
    space := ref AtomSpace;
    space.atoms_by_id = nil;
    space.nodes_by_type_name = nil;
    space.atoms_by_type = nil;
    return space;
}

# Helper to add to list
add_to_atom_list(l: list of (string, ref Atom), key: string, value: ref Atom): list of (string, ref Atom)
{
    return (key, value) :: l;
}

# Helper to add to node list
add_to_node_list(l: list of (string, ref Node), key: string, value: ref Node): list of (string, ref Node)
{
    return (key, value) :: l;
}

AtomSpace.add_node(space: self ref AtomSpace, atomtype: string, name: string): ref Node
{
    # Check if node already exists
    key := atomtype + ":" + name;
    for (l := space.nodes_by_type_name; l != nil; l = tl l) {
        (k, node) := hd l;
        if (k == key)
            return node;
    }
    
    # Create new node
    node := Node.mk(atomtype, name);
    
    # Add to indices
    space.atoms_by_id = add_to_atom_list(space.atoms_by_id, node.atom.id, ref node.atom);
    space.nodes_by_type_name = add_to_node_list(space.nodes_by_type_name, key, node);
    
    return node;
}

AtomSpace.add_link(space: self ref AtomSpace, atomtype: string, outgoing: array of ref Atom): ref Link
{
    # Create new link
    link := Link.mk(atomtype, outgoing);
    
    # Add to indices
    space.atoms_by_id = add_to_atom_list(space.atoms_by_id, link.atom.id, ref link.atom);
    
    return link;
}

AtomSpace.add(space: self ref AtomSpace, atom: ref Atom): ref Atom
{
    # Add atom to the space
    space.atoms_by_id = add_to_atom_list(space.atoms_by_id, atom.id, atom);
    return atom;
}

AtomSpace.remove(space: self ref AtomSpace, atom: ref Atom): int
{
    # Remove from atoms_by_id
    new_list: list of (string, ref Atom) = nil;
    found := 0;
    
    for (l := space.atoms_by_id; l != nil; l = tl l) {
        (k, a) := hd l;
        if (k != atom.id)
            new_list = (k, a) :: new_list;
        else
            found = 1;
    }
    
    space.atoms_by_id = new_list;
    return found;
}

AtomSpace.get_atom(space: self ref AtomSpace, id: string): ref Atom
{
    for (l := space.atoms_by_id; l != nil; l = tl l) {
        (k, atom) := hd l;
        if (k == id)
            return atom;
    }
    return nil;
}

AtomSpace.get_node(space: self ref AtomSpace, atomtype: string, name: string): ref Node
{
    key := atomtype + ":" + name;
    for (l := space.nodes_by_type_name; l != nil; l = tl l) {
        (k, node) := hd l;
        if (k == key)
            return node;
    }
    return nil;
}

AtomSpace.get_atoms_by_type(space: self ref AtomSpace, atomtype: string): list of ref Atom
{
    result: list of ref Atom = nil;
    
    for (l := space.atoms_by_id; l != nil; l = tl l) {
        (k, atom) := hd l;
        if (atom.atomtype == atomtype)
            result = atom :: result;
    }
    
    return result;
}

AtomSpace.query(space: self ref AtomSpace, pattern: ref Atom): list of ref Atom
{
    # Simple pattern matching - returns atoms matching the pattern
    result: list of ref Atom = nil;
    
    # For now, just return atoms of the same type
    # In a full implementation, this would do sophisticated pattern matching
    for (l := space.atoms_by_id; l != nil; l = tl l) {
        (k, atom) := hd l;
        if (atom.atomtype == pattern.atomtype)
            result = atom :: result;
    }
    
    return result;
}

AtomSpace.size(space: self ref AtomSpace): int
{
    count := 0;
    for (l := space.atoms_by_id; l != nil; l = tl l)
        count++;
    return count;
}
