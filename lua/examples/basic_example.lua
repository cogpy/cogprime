#!/usr/bin/env lua5.3
-- OpenCog Example: Basic Usage

-- Add the parent directory to the package path
package.path = package.path .. ";../?.lua;?.lua"

local opencog = require('opencog.init')

print("=== OpenCog in Pure Lua - Basic Example ===\n")

-- Create an AtomSpace
local atomspace = opencog.create_atomspace()
print("Created AtomSpace")

-- Create some nodes
local cat = opencog.create_node("ConceptNode", "cat")
local animal = opencog.create_node("ConceptNode", "animal")
local mammal = opencog.create_node("ConceptNode", "mammal")
local pet = opencog.create_node("ConceptNode", "pet")

print("\nCreated nodes:")
print("  " .. tostring(cat))
print("  " .. tostring(animal))
print("  " .. tostring(mammal))
print("  " .. tostring(pet))

-- Add nodes to atomspace
atomspace:add(cat)
atomspace:add(animal)
atomspace:add(mammal)
atomspace:add(pet)

-- Create inheritance links
local cat_is_mammal = opencog.create_link("InheritanceLink", {cat, mammal})
local mammal_is_animal = opencog.create_link("InheritanceLink", {mammal, animal})
local cat_is_pet = opencog.create_link("InheritanceLink", {cat, pet})

-- Set truth values
cat_is_mammal:set_truth_value(opencog.TruthValue.new(0.95, 0.9))
mammal_is_animal:set_truth_value(opencog.TruthValue.new(1.0, 1.0))
cat_is_pet:set_truth_value(opencog.TruthValue.new(0.7, 0.8))

print("\nCreated inheritance links:")
print("  " .. tostring(cat_is_mammal) .. " " .. tostring(cat_is_mammal:get_truth_value()))
print("  " .. tostring(mammal_is_animal) .. " " .. tostring(mammal_is_animal:get_truth_value()))
print("  " .. tostring(cat_is_pet) .. " " .. tostring(cat_is_pet:get_truth_value()))

-- Add links to atomspace
atomspace:add(cat_is_mammal)
atomspace:add(mammal_is_animal)
atomspace:add(cat_is_pet)

print("\nAtomSpace size: " .. atomspace:size())

-- Pattern matching example
print("\n=== Pattern Matching ===")
local pattern_matcher = opencog.PatternMatcher.new(atomspace)

-- Find all inheritance links
local inheritance_pattern = opencog.create_link("InheritanceLink", {nil, nil})
local matches = atomspace:query(inheritance_pattern)

print("Found " .. #matches .. " inheritance links:")
for _, match in ipairs(matches) do
    print("  " .. tostring(match))
end

-- Reasoning example
print("\n=== Reasoning with PLN ===")
local reasoner = opencog.Reasoner.new(atomspace)

-- Perform deduction: cat->mammal, mammal->animal => cat->animal
local inference = reasoner:deduction(cat_is_mammal, mammal_is_animal)
if inference then
    print("Deduction result:")
    print("  " .. tostring(inference))
    print("  " .. tostring(inference:get_truth_value()))
    atomspace:add(inference)
end

-- Forward chaining
print("\nPerforming forward chaining...")
local new_inferences = reasoner:forward_chain(3)
print("Generated " .. #new_inferences .. " new inferences")

-- Attention allocation example
print("\n=== Attention Allocation ===")
local attention_bank = opencog.AttentionBank.new(atomspace, {
    total_sti = 1000.0,
    sti_threshold = 50.0
})

-- Stimulate important atoms
attention_bank:stimulate(cat, 100)
attention_bank:stimulate(mammal, 80)
attention_bank:stimulate(animal, 60)

print("Stimulated atoms")

-- Update attention
attention_bank:update_attention()

-- Get attentional focus
local focused = attention_bank:get_attentional_focus()
print("\nAtoms in attentional focus (" .. #focused .. " atoms):")
for i, atom in ipairs(focused) do
    if i <= 3 then  -- Show top 3
        local av = atom:get_attention_value()
        print(string.format("  %s - STI: %.2f", tostring(atom), av.sti))
    end
end

-- Learning example
print("\n=== Learning ===")
local learning = opencog.Learning.new(atomspace)

-- Mine frequent patterns
local patterns = learning:mine_frequent_patterns(1)
print("Found " .. #patterns .. " patterns:")
for i, pattern_info in ipairs(patterns) do
    if i <= 3 then  -- Show top 3
        print(string.format("  %s (support: %d)", pattern_info.pattern, pattern_info.support))
    end
end

-- Reinforce important knowledge
print("\nReinforcing cat knowledge...")
learning:reinforce_atom(cat, 50)

print("\n=== Summary ===")
print("AtomSpace final size: " .. atomspace:size())
print("Top atoms by importance:")
local top_atoms = attention_bank:get_top_atoms(3)
for i, atom in ipairs(top_atoms) do
    local av = atom:get_attention_value()
    print(string.format("  %d. %s - STI: %.2f", i, tostring(atom), av.sti))
end

print("\n=== OpenCog Example Complete ===")
