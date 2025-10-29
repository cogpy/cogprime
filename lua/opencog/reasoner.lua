-- OpenCog PLN (Probabilistic Logic Networks) in Pure Lua
-- Inference engine for reasoning

local Reasoner = {}
Reasoner.__index = Reasoner

-- PLN inference confidence factors
local DEDUCTION_CONFIDENCE_FACTOR = 0.9
local INDUCTION_CONFIDENCE_FACTOR = 0.8
local ABDUCTION_CONFIDENCE_FACTOR = 0.7

function Reasoner.new(atomspace)
    local self = setmetatable({}, Reasoner)
    self.atomspace = atomspace
    return self
end

-- Deduction: From A->B and B->C, infer A->C
function Reasoner:deduction(inheritance_ab, inheritance_bc)
    local Node = require('opencog.node')
    local Link = require('opencog.link')
    local TruthValue = require('opencog.atom').TruthValue
    
    if inheritance_ab.atom_type ~= "InheritanceLink" or
       inheritance_bc.atom_type ~= "InheritanceLink" then
        return nil
    end
    
    -- Check if B matches
    local a = inheritance_ab.outgoing_set[1]
    local b1 = inheritance_ab.outgoing_set[2]
    local b2 = inheritance_bc.outgoing_set[1]
    local c = inheritance_bc.outgoing_set[2]
    
    if b1.name ~= b2.name then
        return nil
    end
    
    -- Calculate truth value
    local tv_ab = inheritance_ab:get_truth_value()
    local tv_bc = inheritance_bc:get_truth_value()
    
    -- PLN deduction formula (simplified)
    local strength = tv_ab.strength * tv_bc.strength
    local confidence = tv_ab.confidence * tv_bc.confidence * DEDUCTION_CONFIDENCE_FACTOR
    
    -- Create new inference
    local inference = Link.new("InheritanceLink", {a, c})
    inference:set_truth_value(TruthValue.new(strength, confidence))
    
    return inference
end

-- Induction: From A->B and A->C, infer B->C
function Reasoner:induction(inheritance_ab, inheritance_ac)
    local Link = require('opencog.link')
    local TruthValue = require('opencog.atom').TruthValue
    
    if inheritance_ab.atom_type ~= "InheritanceLink" or
       inheritance_ac.atom_type ~= "InheritanceLink" then
        return nil
    end
    
    -- Check if A matches
    local a1 = inheritance_ab.outgoing_set[1]
    local b = inheritance_ab.outgoing_set[2]
    local a2 = inheritance_ac.outgoing_set[1]
    local c = inheritance_ac.outgoing_set[2]
    
    if a1.name ~= a2.name then
        return nil
    end
    
    -- Calculate truth value
    local tv_ab = inheritance_ab:get_truth_value()
    local tv_ac = inheritance_ac:get_truth_value()
    
    -- PLN induction formula (simplified)
    local strength = tv_ab.strength * tv_ac.strength
    local confidence = tv_ab.confidence * tv_ac.confidence * INDUCTION_CONFIDENCE_FACTOR
    
    -- Create new inference
    local inference = Link.new("InheritanceLink", {b, c})
    inference:set_truth_value(TruthValue.new(strength, confidence))
    
    return inference
end

-- Abduction: From B->C and A->C, infer A->B
function Reasoner:abduction(inheritance_bc, inheritance_ac)
    local Link = require('opencog.link')
    local TruthValue = require('opencog.atom').TruthValue
    
    if inheritance_bc.atom_type ~= "InheritanceLink" or
       inheritance_ac.atom_type ~= "InheritanceLink" then
        return nil
    end
    
    -- Check if C matches
    local b = inheritance_bc.outgoing_set[1]
    local c1 = inheritance_bc.outgoing_set[2]
    local a = inheritance_ac.outgoing_set[1]
    local c2 = inheritance_ac.outgoing_set[2]
    
    if c1.name ~= c2.name then
        return nil
    end
    
    -- Calculate truth value
    local tv_bc = inheritance_bc:get_truth_value()
    local tv_ac = inheritance_ac:get_truth_value()
    
    -- PLN abduction formula (simplified)
    local strength = tv_bc.strength * tv_ac.strength
    local confidence = tv_bc.confidence * tv_ac.confidence * ABDUCTION_CONFIDENCE_FACTOR
    
    -- Create new inference
    local inference = Link.new("InheritanceLink", {a, b})
    inference:set_truth_value(TruthValue.new(strength, confidence))
    
    return inference
end

-- Revision: Combine two truth values for the same statement
function Reasoner:revision(tv1, tv2)
    local TruthValue = require('opencog.atom').TruthValue
    
    -- PLN revision formula
    local w1 = tv1.confidence
    local w2 = tv2.confidence
    local total_weight = w1 + w2
    
    if total_weight == 0 then
        return TruthValue.new(0.5, 0.0)
    end
    
    local strength = (tv1.strength * w1 + tv2.strength * w2) / total_weight
    local confidence = total_weight / (total_weight + 1)
    
    return TruthValue.new(strength, confidence)
end

-- Perform forward chaining inference
function Reasoner:forward_chain(max_steps)
    max_steps = max_steps or 10
    local new_inferences = {}
    
    for step = 1, max_steps do
        local inheritance_links = self.atomspace:get_atoms_by_type("InheritanceLink")
        local step_inferences = {}
        
        -- Try deduction for all pairs
        for i = 1, #inheritance_links do
            for j = 1, #inheritance_links do
                if i ~= j then
                    local inference = self:deduction(inheritance_links[i], inheritance_links[j])
                    if inference then
                        table.insert(step_inferences, inference)
                    end
                end
            end
        end
        
        -- Add new inferences to atomspace
        for _, inference in ipairs(step_inferences) do
            -- Check if already exists
            local existing = self.atomspace:query(inference)
            if #existing == 0 then
                self.atomspace:add(inference)
                table.insert(new_inferences, inference)
            end
        end
        
        -- Stop if no new inferences
        if #step_inferences == 0 then
            break
        end
    end
    
    return new_inferences
end

-- Backward chaining to prove a goal
function Reasoner:backward_chain(goal, max_depth)
    max_depth = max_depth or 5
    
    local function prove(current_goal, depth)
        if depth <= 0 then
            return false
        end
        
        -- Check if goal is directly in atomspace
        local matches = self.atomspace:query(current_goal)
        if #matches > 0 then
            return true
        end
        
        -- Try to prove via deduction
        if current_goal.atom_type == "InheritanceLink" then
            local a = current_goal.outgoing_set[1]
            local c = current_goal.outgoing_set[2]
            
            -- Find intermediate B such that A->B and B->C exist
            local all_inheritance = self.atomspace:get_atoms_by_type("InheritanceLink")
            
            for _, link in ipairs(all_inheritance) do
                if link.outgoing_set[1].name == a.name then
                    local b = link.outgoing_set[2]
                    
                    -- Try to prove B->C
                    local Link = require('opencog.link')
                    local bc_goal = Link.new("InheritanceLink", {b, c})
                    
                    if prove(bc_goal, depth - 1) then
                        return true
                    end
                end
            end
        end
        
        return false
    end
    
    return prove(goal, max_depth)
end

return Reasoner
