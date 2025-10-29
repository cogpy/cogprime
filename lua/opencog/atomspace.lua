-- OpenCog AtomSpace Implementation in Pure Lua
-- Main AtomSpace class for managing atoms and performing operations

local Node = require('opencog.node')
local Link = require('opencog.link')

local AtomSpace = {}
AtomSpace.__index = AtomSpace

function AtomSpace.new()
    local self = setmetatable({}, AtomSpace)
    self.atoms_by_id = {}           -- id -> atom
    self.nodes_by_type_name = {}    -- type:name -> node
    self.atoms_by_type = {}         -- type -> list of atoms
    self.event_handlers = {}        -- event_name -> list of handlers
    return self
end

-- Add an atom to the AtomSpace
function AtomSpace:add(atom)
    if not atom then
        error("Cannot add nil atom")
    end
    
    -- Set the atomspace reference
    atom.atomspace = self
    
    -- Add to main index
    self.atoms_by_id[atom.id] = atom
    
    -- Index by type
    if not self.atoms_by_type[atom.atom_type] then
        self.atoms_by_type[atom.atom_type] = {}
    end
    table.insert(self.atoms_by_type[atom.atom_type], atom)
    
    -- Index nodes by type and name
    if atom:is_node() then
        local key = atom.atom_type .. ":" .. atom.name
        self.nodes_by_type_name[key] = atom
    end
    
    -- Update incoming sets for atoms in the outgoing set
    if atom:is_link() then
        for _, outgoing_atom in ipairs(atom.outgoing_set) do
            outgoing_atom:add_to_incoming_set(atom)
        end
    end
    
    -- Trigger events
    self:trigger_event('atom_added', atom)
    
    return atom
end

-- Remove an atom from the AtomSpace
function AtomSpace:remove(atom)
    if not atom or not self.atoms_by_id[atom.id] then
        return false
    end
    
    -- Remove from type index
    if self.atoms_by_type[atom.atom_type] then
        for i, a in ipairs(self.atoms_by_type[atom.atom_type]) do
            if a.id == atom.id then
                table.remove(self.atoms_by_type[atom.atom_type], i)
                break
            end
        end
        if #self.atoms_by_type[atom.atom_type] == 0 then
            self.atoms_by_type[atom.atom_type] = nil
        end
    end
    
    -- Remove from type-name index if it's a node
    if atom:is_node() then
        local key = atom.atom_type .. ":" .. atom.name
        self.nodes_by_type_name[key] = nil
    end
    
    -- Update incoming sets for atoms in the outgoing set
    if atom:is_link() then
        for _, outgoing_atom in ipairs(atom.outgoing_set) do
            outgoing_atom:remove_from_incoming_set(atom)
        end
    end
    
    -- Remove from main index
    self.atoms_by_id[atom.id] = nil
    
    -- Clear the atomspace reference
    atom.atomspace = nil
    
    -- Trigger events
    self:trigger_event('atom_removed', atom)
    
    return true
end

-- Get an atom by ID
function AtomSpace:get_atom(atom_id)
    return self.atoms_by_id[atom_id]
end

-- Get a node by type and name
function AtomSpace:get_node(atom_type, name)
    local key = atom_type .. ":" .. name
    return self.nodes_by_type_name[key]
end

-- Get all atoms of a given type
function AtomSpace:get_atoms_by_type(atom_type)
    return self.atoms_by_type[atom_type] or {}
end

-- Query atoms matching a pattern
function AtomSpace:query(pattern)
    local results = {}
    local atom_module = require('opencog.atom')
    local Wildcard = atom_module.Wildcard
    
    -- If pattern is a node, look it up directly
    if pattern:is_node() then
        local atom = self:get_node(pattern.atom_type, pattern.name)
        if atom then
            table.insert(results, atom)
        end
        return results
    end
    
    -- If pattern is a link, find matching links
    if pattern:is_link() then
        local candidates = self:get_atoms_by_type(pattern.atom_type)
        
        for _, candidate in ipairs(candidates) do
            if candidate:is_link() then
                -- If pattern has no outgoing set (all wildcards), match all candidates
                if #pattern.outgoing_set == 0 then
                    table.insert(results, candidate)
                -- Check if outgoing sets match
                elseif #candidate.outgoing_set == #pattern.outgoing_set then
                    local match = true
                    
                    for i, pattern_atom in ipairs(pattern.outgoing_set) do
                        local candidate_atom = candidate.outgoing_set[i]
                        
                        -- Check if it's a wildcard
                        local is_wildcard = false
                        if type(pattern_atom) == "table" and pattern_atom._is_wildcard then
                            is_wildcard = true
                        end
                        
                        -- nil or wildcard is a match
                        if not is_wildcard and pattern_atom ~= nil then
                            -- Check if atoms match
                            if pattern_atom:is_node() and candidate_atom:is_node() then
                                if pattern_atom.atom_type ~= candidate_atom.atom_type or
                                   pattern_atom.name ~= candidate_atom.name then
                                    match = false
                                    break
                                end
                            elseif pattern_atom:is_link() and candidate_atom:is_link() then
                                -- Recursive match for nested links
                                if pattern_atom.atom_type ~= candidate_atom.atom_type then
                                    match = false
                                    break
                                end
                            else
                                match = false
                                break
                            end
                        end
                    end
                    
                    if match then
                        table.insert(results, candidate)
                    end
                end
            end
        end
    end
    
    return results
end

-- Get all atoms in the atomspace
function AtomSpace:get_all_atoms()
    local atoms = {}
    for _, atom in pairs(self.atoms_by_id) do
        table.insert(atoms, atom)
    end
    return atoms
end

-- Get the size of the atomspace
function AtomSpace:size()
    local count = 0
    for _ in pairs(self.atoms_by_id) do
        count = count + 1
    end
    return count
end

-- Register an event handler
function AtomSpace:register_event_handler(event_name, handler)
    if not self.event_handlers[event_name] then
        self.event_handlers[event_name] = {}
    end
    table.insert(self.event_handlers[event_name], handler)
end

-- Unregister an event handler
function AtomSpace:unregister_event_handler(event_name, handler)
    if self.event_handlers[event_name] then
        for i, h in ipairs(self.event_handlers[event_name]) do
            if h == handler then
                table.remove(self.event_handlers[event_name], i)
                return true
            end
        end
    end
    return false
end

-- Trigger an event
function AtomSpace:trigger_event(event_name, ...)
    if self.event_handlers[event_name] then
        for _, handler in ipairs(self.event_handlers[event_name]) do
            local success, err = pcall(handler, ...)
            if not success then
                print(string.format("Error in event handler for %s: %s", event_name, err))
            end
        end
    end
end

-- Clear all atoms from the atomspace
function AtomSpace:clear()
    self.atoms_by_id = {}
    self.nodes_by_type_name = {}
    self.atoms_by_type = {}
end

return AtomSpace
