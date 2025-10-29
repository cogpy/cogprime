-- OpenCog Pattern Matcher in Pure Lua
-- Advanced pattern matching for the AtomSpace

local PatternMatcher = {}
PatternMatcher.__index = PatternMatcher

function PatternMatcher.new(atomspace)
    local self = setmetatable({}, PatternMatcher)
    self.atomspace = atomspace
    return self
end

-- Match a variable pattern against an atom
local function match_variable(var_name, atom, bindings)
    if bindings[var_name] then
        -- Variable already bound, check if it matches
        return bindings[var_name].id == atom.id
    else
        -- Bind the variable
        bindings[var_name] = atom
        return true
    end
end

-- Match a pattern atom against a candidate atom
local function match_atom(pattern, candidate, bindings)
    -- Handle variable patterns (starting with $)
    if type(pattern) == "string" and pattern:sub(1, 1) == "$" then
        return match_variable(pattern, candidate, bindings)
    end
    
    -- Handle nil (wildcard)
    if pattern == nil then
        return true
    end
    
    -- Handle node patterns
    if pattern:is_node() and candidate:is_node() then
        return pattern.atom_type == candidate.atom_type and
               pattern.name == candidate.name
    end
    
    -- Handle link patterns
    if pattern:is_link() and candidate:is_link() then
        if pattern.atom_type ~= candidate.atom_type then
            return false
        end
        
        if #pattern.outgoing_set ~= #candidate.outgoing_set then
            return false
        end
        
        -- Recursively match outgoing sets
        for i = 1, #pattern.outgoing_set do
            if not match_atom(pattern.outgoing_set[i], 
                            candidate.outgoing_set[i], bindings) then
                return false
            end
        end
        
        return true
    end
    
    return false
end

-- Find all matches for a pattern in the atomspace
function PatternMatcher:find_matches(pattern)
    local results = {}
    local all_atoms = self.atomspace:get_all_atoms()
    
    for _, atom in ipairs(all_atoms) do
        local bindings = {}
        if match_atom(pattern, atom, bindings) then
            table.insert(results, {atom = atom, bindings = bindings})
        end
    end
    
    return results
end

-- Execute a query with variable bindings
function PatternMatcher:query(pattern, callback)
    local matches = self:find_matches(pattern)
    
    if callback then
        for _, match in ipairs(matches) do
            callback(match.atom, match.bindings)
        end
    end
    
    return matches
end

-- Pattern match with constraints
function PatternMatcher:match_with_constraints(pattern, constraints)
    local matches = self:find_matches(pattern)
    local filtered = {}
    
    for _, match in ipairs(matches) do
        local valid = true
        
        -- Check constraints
        for _, constraint in ipairs(constraints) do
            if not constraint(match.atom, match.bindings) then
                valid = false
                break
            end
        end
        
        if valid then
            table.insert(filtered, match)
        end
    end
    
    return filtered
end

-- Bind pattern - advanced pattern matching with variable substitution
function PatternMatcher:bind_pattern(pattern_link, implicand_link)
    local matches = self:find_matches(pattern_link)
    local results = {}
    
    for _, match in ipairs(matches) do
        -- Substitute variables in implicand with bindings
        local result = self:substitute(implicand_link, match.bindings)
        table.insert(results, result)
    end
    
    return results
end

-- Substitute variables in a pattern with bindings
function PatternMatcher:substitute(pattern, bindings)
    if type(pattern) == "string" and pattern:sub(1, 1) == "$" then
        return bindings[pattern] or pattern
    end
    
    if pattern:is_node() then
        return pattern
    end
    
    if pattern:is_link() then
        local Link = require('opencog.link')
        local new_outgoing = {}
        
        for _, atom in ipairs(pattern.outgoing_set) do
            table.insert(new_outgoing, self:substitute(atom, bindings))
        end
        
        return Link.new(pattern.atom_type, new_outgoing)
    end
    
    return pattern
end

return PatternMatcher
