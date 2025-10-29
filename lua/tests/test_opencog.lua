#!/usr/bin/env lua5.3
-- Test suite for OpenCog in Pure Lua

package.path = package.path .. ";../?.lua;?.lua"

local opencog = require('opencog.init')

local tests_passed = 0
local tests_failed = 0

-- Test helper function
local function test(name, func)
    io.write("Testing " .. name .. "... ")
    local success, err = pcall(func)
    if success then
        io.write("PASSED\n")
        tests_passed = tests_passed + 1
    else
        io.write("FAILED: " .. tostring(err) .. "\n")
        tests_failed = tests_failed + 1
    end
end

-- Assert helper
local function assert_equal(actual, expected, msg)
    if actual ~= expected then
        error(msg or string.format("Expected %s but got %s", tostring(expected), tostring(actual)))
    end
end

local function assert_true(value, msg)
    if not value then
        error(msg or "Expected true but got false")
    end
end

print("=== OpenCog Lua Test Suite ===\n")

-- Test 1: AtomSpace creation
test("AtomSpace creation", function()
    local atomspace = opencog.create_atomspace()
    assert_true(atomspace ~= nil, "AtomSpace should not be nil")
    assert_equal(atomspace:size(), 0, "New atomspace should be empty")
end)

-- Test 2: Node creation
test("Node creation", function()
    local node = opencog.create_node("ConceptNode", "test")
    assert_true(node ~= nil, "Node should not be nil")
    assert_equal(node.atom_type, "ConceptNode")
    assert_equal(node.name, "test")
    assert_true(node:is_node())
    assert_true(not node:is_link())
end)

-- Test 3: Link creation
test("Link creation", function()
    local node1 = opencog.create_node("ConceptNode", "A")
    local node2 = opencog.create_node("ConceptNode", "B")
    local link = opencog.create_link("InheritanceLink", {node1, node2})
    assert_true(link ~= nil, "Link should not be nil")
    assert_equal(link.atom_type, "InheritanceLink")
    assert_true(link:is_link())
    assert_true(not link:is_node())
    assert_equal(link:get_arity(), 2)
end)

-- Test 4: Adding atoms to atomspace
test("Adding atoms to atomspace", function()
    local atomspace = opencog.create_atomspace()
    local node = opencog.create_node("ConceptNode", "test")
    atomspace:add(node)
    assert_equal(atomspace:size(), 1)
end)

-- Test 5: Retrieving nodes
test("Retrieving nodes", function()
    local atomspace = opencog.create_atomspace()
    local node = opencog.create_node("ConceptNode", "test")
    atomspace:add(node)
    local retrieved = atomspace:get_node("ConceptNode", "test")
    assert_true(retrieved ~= nil)
    assert_equal(retrieved.name, "test")
end)

-- Test 6: TruthValue
test("TruthValue", function()
    local tv = opencog.TruthValue.new(0.8, 0.9)
    assert_equal(tv.strength, 0.8)
    assert_equal(tv.confidence, 0.9)
    
    -- Test clamping
    local tv2 = opencog.TruthValue.new(1.5, -0.5)
    assert_equal(tv2.strength, 1.0)
    assert_equal(tv2.confidence, 0.0)
end)

-- Test 7: AttentionValue
test("AttentionValue", function()
    local av = opencog.AttentionValue.new(100, 50, false)
    assert_equal(av.sti, 100)
    assert_equal(av.lti, 50)
    assert_equal(av.vlti, false)
end)

-- Test 8: Atom truth and attention values
test("Atom truth and attention values", function()
    local node = opencog.create_node("ConceptNode", "test")
    local tv = opencog.TruthValue.new(0.7, 0.8)
    node:set_truth_value(tv)
    local retrieved_tv = node:get_truth_value()
    assert_equal(retrieved_tv.strength, 0.7)
    assert_equal(retrieved_tv.confidence, 0.8)
end)

-- Test 9: Pattern matching
test("Pattern matching", function()
    local atomspace = opencog.create_atomspace()
    local cat = opencog.create_node("ConceptNode", "cat")
    local animal = opencog.create_node("ConceptNode", "animal")
    local link = opencog.create_link("InheritanceLink", {cat, animal})
    
    atomspace:add(cat)
    atomspace:add(animal)
    atomspace:add(link)
    
    -- Query for the specific link
    local pattern = opencog.create_link("InheritanceLink", {cat, animal})
    local matches = atomspace:query(pattern)
    assert_equal(#matches, 1)
end)

-- Test 10: Wildcard pattern matching
test("Wildcard pattern matching", function()
    local atomspace = opencog.create_atomspace()
    local cat = opencog.create_node("ConceptNode", "cat")
    local dog = opencog.create_node("ConceptNode", "dog")
    local animal = opencog.create_node("ConceptNode", "animal")
    
    atomspace:add(cat)
    atomspace:add(dog)
    atomspace:add(animal)
    atomspace:add(opencog.create_link("InheritanceLink", {cat, animal}))
    atomspace:add(opencog.create_link("InheritanceLink", {dog, animal}))
    
    -- Query for all inheritance links
    local pattern = opencog.create_link("InheritanceLink", {nil, nil})
    local matches = atomspace:query(pattern)
    assert_equal(#matches, 2)
end)

-- Test 11: Reasoning - Deduction
test("Reasoning - Deduction", function()
    local atomspace = opencog.create_atomspace()
    local reasoner = opencog.Reasoner.new(atomspace)
    
    local a = opencog.create_node("ConceptNode", "A")
    local b = opencog.create_node("ConceptNode", "B")
    local c = opencog.create_node("ConceptNode", "C")
    
    local ab = opencog.create_link("InheritanceLink", {a, b})
    local bc = opencog.create_link("InheritanceLink", {b, c})
    
    ab:set_truth_value(opencog.TruthValue.new(0.9, 0.9))
    bc:set_truth_value(opencog.TruthValue.new(0.9, 0.9))
    
    local inference = reasoner:deduction(ab, bc)
    assert_true(inference ~= nil)
    assert_equal(inference.atom_type, "InheritanceLink")
    assert_equal(inference.outgoing_set[1].name, "A")
    assert_equal(inference.outgoing_set[2].name, "C")
end)

-- Test 12: Attention Bank
test("Attention Bank", function()
    local atomspace = opencog.create_atomspace()
    local attention_bank = opencog.AttentionBank.new(atomspace, {
        total_sti = 1000.0,
        sti_threshold = 50.0
    })
    
    local node = opencog.create_node("ConceptNode", "test")
    atomspace:add(node)
    
    attention_bank:stimulate(node, 100)
    local av = node:get_attention_value()
    assert_true(av.sti >= 100)
end)

-- Test 13: Learning - Pattern Mining
test("Learning - Pattern Mining", function()
    local atomspace = opencog.create_atomspace()
    local learning = opencog.Learning.new(atomspace)
    
    local a = opencog.create_node("ConceptNode", "A")
    local b = opencog.create_node("ConceptNode", "B")
    
    atomspace:add(a)
    atomspace:add(b)
    atomspace:add(opencog.create_link("InheritanceLink", {a, b}))
    atomspace:add(opencog.create_link("InheritanceLink", {b, a}))
    
    local patterns = learning:mine_frequent_patterns(1)
    assert_true(#patterns > 0)
end)

-- Test 14: Removing atoms
test("Removing atoms", function()
    local atomspace = opencog.create_atomspace()
    local node = opencog.create_node("ConceptNode", "test")
    atomspace:add(node)
    assert_equal(atomspace:size(), 1)
    
    local removed = atomspace:remove(node)
    assert_true(removed)
    assert_equal(atomspace:size(), 0)
end)

-- Test 15: Incoming set
test("Incoming set", function()
    local atomspace = opencog.create_atomspace()
    local a = opencog.create_node("ConceptNode", "A")
    local b = opencog.create_node("ConceptNode", "B")
    local link = opencog.create_link("InheritanceLink", {a, b})
    
    atomspace:add(a)
    atomspace:add(b)
    atomspace:add(link)
    
    local incoming = a:get_incoming_set()
    assert_equal(#incoming, 1)
    assert_equal(incoming[1].atom_type, "InheritanceLink")
end)

-- Summary
print("\n=== Test Results ===")
print(string.format("Passed: %d", tests_passed))
print(string.format("Failed: %d", tests_failed))
print(string.format("Total: %d", tests_passed + tests_failed))

if tests_failed == 0 then
    print("\n✓ All tests passed!")
    os.exit(0)
else
    print("\n✗ Some tests failed")
    os.exit(1)
end
