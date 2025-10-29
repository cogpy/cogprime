-- OpenCog Learning Module in Pure Lua
-- Pattern mining and learning capabilities

local Learning = {}
Learning.__index = Learning

-- Learning configuration constants
local LTI_DECAY_RATE = 0.9
local LTI_UPDATE_RATE = 0.1
local OUTGOING_REWARD_PROPAGATION_RATE = 0.5
local INCOMING_REWARD_PROPAGATION_RATE = 0.3
local MIN_REWARD_THRESHOLD = 0.1

function Learning.new(atomspace)
    local self = setmetatable({}, Learning)
    self.atomspace = atomspace
    self.pattern_cache = {}
    return self
end

-- Mine frequent patterns from the atomspace
function Learning:mine_frequent_patterns(min_support)
    min_support = min_support or 2
    local patterns = {}
    local atoms = self.atomspace:get_all_atoms()
    
    -- Count pattern occurrences
    local pattern_counts = {}
    
    for _, atom in ipairs(atoms) do
        if atom:is_link() then
            local pattern_key = self:get_pattern_signature(atom)
            pattern_counts[pattern_key] = (pattern_counts[pattern_key] or 0) + 1
        end
    end
    
    -- Filter by minimum support
    for pattern_key, count in pairs(pattern_counts) do
        if count >= min_support then
            table.insert(patterns, {
                pattern = pattern_key,
                support = count
            })
        end
    end
    
    -- Sort by support (descending)
    table.sort(patterns, function(a, b)
        return a.support > b.support
    end)
    
    return patterns
end

-- Get a pattern signature for an atom
function Learning:get_pattern_signature(atom)
    if atom:is_node() then
        return atom.atom_type
    end
    
    if atom:is_link() then
        local parts = {atom.atom_type}
        for _, outgoing in ipairs(atom.outgoing_set) do
            table.insert(parts, self:get_pattern_signature(outgoing))
        end
        return table.concat(parts, "->")
    end
    
    return "Unknown"
end

-- Learn correlations between atoms
function Learning:learn_correlations(threshold)
    threshold = threshold or 0.5
    local correlations = {}
    local atoms = self.atomspace:get_all_atoms()
    
    -- Find co-occurring atoms in links
    for _, atom in ipairs(atoms) do
        if atom:is_link() then
            local outgoing = atom.outgoing_set
            
            -- Check all pairs in the outgoing set
            for i = 1, #outgoing do
                for j = i + 1, #outgoing do
                    local pair_key = outgoing[i].id .. ":" .. outgoing[j].id
                    correlations[pair_key] = (correlations[pair_key] or 0) + 1
                end
            end
        end
    end
    
    -- Normalize and filter
    local total_links = #self.atomspace:get_atoms_by_type("Link") or 1
    local filtered = {}
    
    for pair_key, count in pairs(correlations) do
        local correlation_strength = count / total_links
        if correlation_strength >= threshold then
            table.insert(filtered, {
                pair = pair_key,
                strength = correlation_strength
            })
        end
    end
    
    return filtered
end

-- Reinforcement learning for atom importance
function Learning:reinforce_atom(atom, reward, visited)
    visited = visited or {}
    
    -- Avoid cycles
    if visited[atom.id] then
        return
    end
    visited[atom.id] = true
    
    local av = atom:get_attention_value()
    
    -- Update STI based on reward
    av.sti = av.sti + reward
    
    -- Update LTI with exponential moving average
    av.lti = av.lti * LTI_DECAY_RATE + reward * LTI_UPDATE_RATE
    
    atom:set_attention_value(av)
    
    -- Propagate reward to related atoms (with decay)
    if reward > MIN_REWARD_THRESHOLD then  -- Stop if reward is too small
        if atom:is_link() then
            local propagated_reward = reward * OUTGOING_REWARD_PROPAGATION_RATE
            for _, outgoing in ipairs(atom.outgoing_set) do
                self:reinforce_atom(outgoing, propagated_reward, visited)
            end
        end
        
        -- Propagate through incoming set
        for _, incoming in ipairs(atom:get_incoming_set()) do
            local propagated_reward = reward * INCOMING_REWARD_PROPAGATION_RATE
            self:reinforce_atom(incoming, propagated_reward, visited)
        end
    end
end

-- Learn from examples (supervised learning)
function Learning:learn_from_examples(examples)
    local Link = require('opencog.link')
    local TruthValue = require('opencog.atom').TruthValue
    
    for _, example in ipairs(examples) do
        local input_atoms = example.input
        local output_atom = example.output
        local reward = example.reward or 1.0
        
        -- Create association link
        local association = Link.new("AssociativeLink", input_atoms)
        association:set_truth_value(TruthValue.new(reward, 0.9))
        self.atomspace:add(association)
        
        -- Reinforce the atoms
        for _, atom in ipairs(input_atoms) do
            self:reinforce_atom(atom, reward)
        end
        if output_atom then
            self:reinforce_atom(output_atom, reward * 1.5)
        end
    end
end

-- Pattern completion - predict missing parts
function Learning:complete_pattern(partial_pattern)
    local Link = require('opencog.link')
    local candidates = {}
    local all_links = self.atomspace:get_all_atoms()
    
    for _, atom in ipairs(all_links) do
        if atom:is_link() and atom.atom_type == partial_pattern.atom_type then
            -- Check if partial pattern matches
            local matches = true
            for i, pattern_atom in ipairs(partial_pattern.outgoing_set) do
                if pattern_atom ~= nil then
                    if i > #atom.outgoing_set or
                       atom.outgoing_set[i].name ~= pattern_atom.name then
                        matches = false
                        break
                    end
                end
            end
            
            if matches then
                table.insert(candidates, atom)
            end
        end
    end
    
    -- Sort by importance (STI)
    table.sort(candidates, function(a, b)
        return a:get_attention_value().sti > b:get_attention_value().sti
    end)
    
    return candidates
end

-- Concept formation - cluster similar atoms
function Learning:form_concepts(atoms, similarity_threshold)
    similarity_threshold = similarity_threshold or 0.7
    local concepts = {}
    local assigned = {}
    
    for i, atom1 in ipairs(atoms) do
        if not assigned[atom1.id] then
            local concept = {atom1}
            assigned[atom1.id] = true
            
            for j = i + 1, #atoms do
                local atom2 = atoms[j]
                if not assigned[atom2.id] then
                    local similarity = self:compute_similarity(atom1, atom2)
                    if similarity >= similarity_threshold then
                        table.insert(concept, atom2)
                        assigned[atom2.id] = true
                    end
                end
            end
            
            if #concept > 1 then
                table.insert(concepts, concept)
            end
        end
    end
    
    return concepts
end

-- Compute similarity between two atoms
function Learning:compute_similarity(atom1, atom2)
    -- Type similarity
    if atom1.atom_type ~= atom2.atom_type then
        return 0.0
    end
    
    -- Node similarity (name matching)
    if atom1:is_node() and atom2:is_node() then
        return atom1.name == atom2.name and 1.0 or 0.0
    end
    
    -- Link similarity (structure matching)
    if atom1:is_link() and atom2:is_link() then
        if #atom1.outgoing_set ~= #atom2.outgoing_set then
            return 0.0
        end
        
        local total_similarity = 0.0
        for i = 1, #atom1.outgoing_set do
            total_similarity = total_similarity + 
                self:compute_similarity(atom1.outgoing_set[i], atom2.outgoing_set[i])
        end
        
        return total_similarity / #atom1.outgoing_set
    end
    
    return 0.0
end

return Learning
