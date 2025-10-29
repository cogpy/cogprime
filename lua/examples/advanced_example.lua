#!/usr/bin/env lua5.3
-- OpenCog Advanced Example: Knowledge Base and Reasoning

package.path = package.path .. ";../?.lua;?.lua"

local opencog = require('opencog.init')

print("=== OpenCog Advanced Example: Animal Knowledge Base ===\n")

-- Create an AtomSpace
local atomspace = opencog.create_atomspace()
local reasoner = opencog.Reasoner.new(atomspace)
local learning = opencog.Learning.new(atomspace)
local attention = opencog.AttentionBank.new(atomspace, {
    total_sti = 2000.0,
    sti_threshold = 100.0
})

-- Build a knowledge base about animals
print("Building knowledge base...")

-- Create concepts
local animals = {
    cat = opencog.create_node("ConceptNode", "cat"),
    dog = opencog.create_node("ConceptNode", "dog"),
    bird = opencog.create_node("ConceptNode", "bird"),
    fish = opencog.create_node("ConceptNode", "fish"),
    mammal = opencog.create_node("ConceptNode", "mammal"),
    vertebrate = opencog.create_node("ConceptNode", "vertebrate"),
    animal = opencog.create_node("ConceptNode", "animal"),
    pet = opencog.create_node("ConceptNode", "pet"),
    predator = opencog.create_node("ConceptNode", "predator")
}

-- Add all concepts to atomspace
for name, concept in pairs(animals) do
    atomspace:add(concept)
end

-- Create inheritance hierarchy with truth values
local function add_inheritance(from, to, strength, confidence)
    local link = opencog.create_link("InheritanceLink", {from, to})
    link:set_truth_value(opencog.TruthValue.new(strength, confidence))
    atomspace:add(link)
    return link
end

print("\nAdding inheritance relationships...")

-- Specific to general hierarchy
add_inheritance(animals.cat, animals.mammal, 1.0, 1.0)
add_inheritance(animals.dog, animals.mammal, 1.0, 1.0)
add_inheritance(animals.bird, animals.vertebrate, 1.0, 1.0)
add_inheritance(animals.fish, animals.vertebrate, 1.0, 1.0)

add_inheritance(animals.mammal, animals.vertebrate, 1.0, 1.0)
add_inheritance(animals.vertebrate, animals.animal, 1.0, 1.0)

-- Properties
add_inheritance(animals.cat, animals.pet, 0.8, 0.9)
add_inheritance(animals.dog, animals.pet, 0.9, 0.95)
add_inheritance(animals.cat, animals.predator, 0.7, 0.8)

print("Knowledge base created with " .. atomspace:size() .. " atoms")

-- Perform reasoning
print("\n=== Reasoning ===")

-- Forward chaining to infer new knowledge
print("\nPerforming forward chaining...")
local inferences = reasoner:forward_chain(5)
print("Generated " .. #inferences .. " new inferences:")

for i, inference in ipairs(inferences) do
    if i <= 5 then  -- Show first 5
        print(string.format("  %d. %s [%.3f, %.3f]", 
            i, 
            tostring(inference),
            inference:get_truth_value().strength,
            inference:get_truth_value().confidence))
    end
end

-- Pattern matching queries
print("\n=== Pattern Matching ===")

-- Find all pets
local pet_pattern = opencog.create_link("InheritanceLink", {})
local all_inheritance = atomspace:query(pet_pattern)

print("\nAll inheritance links (" .. #all_inheritance .. " total):")
for i, link in ipairs(all_inheritance) do
    if i <= 8 then
        print("  " .. tostring(link))
    end
end

-- Attention dynamics
print("\n=== Attention Allocation ===")

-- Stimulate frequently accessed concepts
attention:stimulate(animals.cat, 150)
attention:stimulate(animals.dog, 140)
attention:stimulate(animals.mammal, 120)
attention:stimulate(animals.pet, 100)

print("\nStimulated key concepts")

-- Update attention multiple times to see dynamics
for i = 1, 3 do
    attention:update_attention()
end

print("Updated attention dynamics")

-- Get attentional focus
local focused = attention:get_attentional_focus()
print("\nConcepts in attentional focus (" .. #focused .. " atoms):")
for i, atom in ipairs(focused) do
    if i <= 5 then
        local av = atom:get_attention_value()
        print(string.format("  %d. %s - STI: %.2f, LTI: %.2f", 
            i, tostring(atom), av.sti, av.lti))
    end
end

-- Learning from interactions
print("\n=== Learning ===")

-- Simulate learning from observations
print("\nSimulating observations (reinforcement learning)...")

local observations = {
    {animals.cat, 30},
    {animals.dog, 28},
    {animals.pet, 25},
    {animals.mammal, 20}
}

for _, obs in ipairs(observations) do
    learning:reinforce_atom(obs[1], obs[2])
end

print("Reinforced atoms based on observations")

-- Mine patterns
local patterns = learning:mine_frequent_patterns(2)
print("\nFrequent patterns discovered (" .. #patterns .. " total):")
for i, pattern_info in ipairs(patterns) do
    if i <= 5 then
        print(string.format("  %d. %s (support: %d)", 
            i, pattern_info.pattern, pattern_info.support))
    end
end

-- Learn correlations
local correlations = learning:learn_correlations(0.3)
print("\nLearned correlations (" .. #correlations .. " total)")

-- Concept formation
print("\nForming concepts from similar atoms...")
local all_nodes = {}
for _, atom in ipairs(atomspace:get_all_atoms()) do
    if atom:is_node() then
        table.insert(all_nodes, atom)
    end
end

local concepts = learning:form_concepts(all_nodes, 0.5)
print("Formed " .. #concepts .. " concept clusters")

-- Final analysis
print("\n=== Knowledge Base Analysis ===")

print("\nKnowledge base statistics:")
print("  Total atoms: " .. atomspace:size())
print("  Nodes: " .. #atomspace:get_atoms_by_type("ConceptNode"))

local all_links = {}
for _, atom in ipairs(atomspace:get_all_atoms()) do
    if atom:is_link() then
        table.insert(all_links, atom)
    end
end
print("  Links: " .. #all_links)

print("\nTop 5 most important concepts:")
local top = attention:get_top_atoms(5)
for i, atom in ipairs(top) do
    local av = atom:get_attention_value()
    print(string.format("  %d. %s - STI: %.2f, LTI: %.2f", 
        i, tostring(atom), av.sti, av.lti))
end

-- Query specific knowledge
print("\n=== Knowledge Queries ===")

-- What inherits from mammal?
print("\nWhat are mammals?")
local mammal_children = {}
for _, link in ipairs(all_links) do
    if link.atom_type == "InheritanceLink" and 
       link.outgoing_set[2].name == "mammal" then
        table.insert(mammal_children, link.outgoing_set[1].name)
    end
end
print("  " .. table.concat(mammal_children, ", "))

-- What are the properties of cat?
print("\nWhat are the properties of cat?")
local cat_properties = {}
for _, link in ipairs(all_links) do
    if link.atom_type == "InheritanceLink" and 
       link.outgoing_set[1].name == "cat" then
        local tv = link:get_truth_value()
        table.insert(cat_properties, 
            string.format("%s (%.2f)", link.outgoing_set[2].name, tv.strength))
    end
end
print("  " .. table.concat(cat_properties, ", "))

print("\n=== Advanced Example Complete ===")
