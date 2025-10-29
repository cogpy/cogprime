implement TestAtomSpace;

#
# Unit tests for AtomSpace module
#

include "sys.m";
    sys: Sys;
include "../opencog/atomspace.m";
    atomspace: Atomspace;
    AtomSpace, Node, Link, TruthValue, AttentionValue: import atomspace;

# Test results
tests_passed := 0;
tests_failed := 0;

init(nil: ref Draw->Context, args: list of string)
{
    sys = load Sys Sys->PATH;
    atomspace = load Atomspace Atomspace->PATH;
    
    if (atomspace == nil) {
        sys->print("FATAL: Cannot load Atomspace module\n");
        raise "fail:load";
    }
    
    sys->print("AtomSpace Unit Tests\n");
    sys->print("====================\n\n");
    
    # Run tests
    test_atomspace_creation();
    test_node_creation();
    test_link_creation();
    test_truth_value();
    test_attention_value();
    test_query();
    test_size();
    
    # Print results
    sys->print("\n");
    sys->print("Test Results\n");
    sys->print("============\n");
    sys->print("Passed: %d\n", tests_passed);
    sys->print("Failed: %d\n", tests_failed);
    
    if (tests_failed == 0)
        sys->print("\nAll tests passed!\n");
    else
        sys->print("\nSome tests failed.\n");
}

# Test helper
assert(condition: int, test_name: string)
{
    if (condition) {
        sys->print("[PASS] %s\n", test_name);
        tests_passed++;
    } else {
        sys->print("[FAIL] %s\n", test_name);
        tests_failed++;
    }
}

# Test AtomSpace creation
test_atomspace_creation()
{
    sys->print("Testing AtomSpace creation...\n");
    
    space := AtomSpace.mk();
    assert(space != nil, "AtomSpace.mk() returns non-nil");
    assert(space.size() == 0, "New AtomSpace has size 0");
}

# Test node creation
test_node_creation()
{
    sys->print("\nTesting Node creation...\n");
    
    space := AtomSpace.mk();
    
    # Create a node
    cat := space.add_node("ConceptNode", "cat");
    assert(cat != nil, "add_node returns non-nil");
    assert(cat.name == "cat", "Node name is correct");
    assert(cat.atom.atomtype == "ConceptNode", "Node type is correct");
    assert(space.size() == 1, "Size increases after adding node");
    
    # Create same node again - should return existing
    cat2 := space.add_node("ConceptNode", "cat");
    assert(cat2 != nil, "Re-adding same node returns non-nil");
    assert(space.size() == 1, "Size doesn't increase for duplicate node");
    
    # Create different node
    dog := space.add_node("ConceptNode", "dog");
    assert(dog != nil, "Second node returns non-nil");
    assert(space.size() == 2, "Size increases for new node");
}

# Test link creation
test_link_creation()
{
    sys->print("\nTesting Link creation...\n");
    
    space := AtomSpace.mk();
    
    # Create nodes
    cat := space.add_node("ConceptNode", "cat");
    animal := space.add_node("ConceptNode", "animal");
    
    # Create link
    outgoing := array[2] of ref atomspace->Atom;
    outgoing[0] = ref cat.atom;
    outgoing[1] = ref animal.atom;
    link := space.add_link("InheritanceLink", outgoing);
    
    assert(link != nil, "add_link returns non-nil");
    assert(link.atom.atomtype == "InheritanceLink", "Link type is correct");
    assert(link.get_arity() == 2, "Link arity is correct");
    assert(space.size() == 3, "Size includes both nodes and link");
}

# Test TruthValue
test_truth_value()
{
    sys->print("\nTesting TruthValue...\n");
    
    # Create truth value
    tv := TruthValue.mk(0.9, 0.8);
    assert(tv != nil, "TruthValue.mk returns non-nil");
    assert(tv.strength == 0.9, "Strength is correct");
    assert(tv.confidence == 0.8, "Confidence is correct");
    
    # Test clamping
    tv2 := TruthValue.mk(1.5, -0.5);
    assert(tv2.strength == 1.0, "Strength clamped to 1.0");
    assert(tv2.confidence == 0.0, "Confidence clamped to 0.0");
    
    # Test with node
    space := AtomSpace.mk();
    cat := space.add_node("ConceptNode", "cat");
    cat.atom.tv = tv;
    assert(cat.atom.tv.strength == 0.9, "Node truth value set correctly");
}

# Test AttentionValue
test_attention_value()
{
    sys->print("\nTesting AttentionValue...\n");
    
    # Create attention value
    av := AttentionValue.mk(100.0, 50.0, 1);
    assert(av != nil, "AttentionValue.mk returns non-nil");
    assert(av.sti == 100.0, "STI is correct");
    assert(av.lti == 50.0, "LTI is correct");
    assert(av.vlti == 1, "VLTI is correct");
    
    # Test with node
    space := AtomSpace.mk();
    cat := space.add_node("ConceptNode", "cat");
    cat.atom.av = av;
    assert(cat.atom.av.sti == 100.0, "Node attention value set correctly");
}

# Test query
test_query()
{
    sys->print("\nTesting query operations...\n");
    
    space := AtomSpace.mk();
    
    # Create some nodes
    cat := space.add_node("ConceptNode", "cat");
    dog := space.add_node("ConceptNode", "dog");
    runs := space.add_node("PredicateNode", "runs");
    
    # Get by type
    concepts := space.get_atoms_by_type("ConceptNode");
    count := 0;
    for (c := concepts; c != nil; c = tl c)
        count++;
    assert(count == 2, "get_atoms_by_type returns correct count");
    
    # Get node by type and name
    retrieved := space.get_node("ConceptNode", "cat");
    assert(retrieved != nil, "get_node returns non-nil");
    assert(retrieved.name == "cat", "get_node returns correct node");
}

# Test size
test_size()
{
    sys->print("\nTesting size operations...\n");
    
    space := AtomSpace.mk();
    assert(space.size() == 0, "Empty space has size 0");
    
    space.add_node("ConceptNode", "a");
    assert(space.size() == 1, "Size is 1 after one node");
    
    space.add_node("ConceptNode", "b");
    assert(space.size() == 2, "Size is 2 after two nodes");
    
    cat := space.add_node("ConceptNode", "cat");
    dog := space.add_node("ConceptNode", "dog");
    outgoing := array[2] of ref atomspace->Atom;
    outgoing[0] = ref cat.atom;
    outgoing[1] = ref dog.atom;
    space.add_link("InheritanceLink", outgoing);
    
    assert(space.size() == 5, "Size includes both nodes and links");
}
