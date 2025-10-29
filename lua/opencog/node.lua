-- OpenCog Node Implementation in Pure Lua
-- Node class representing vertices in the AtomSpace hypergraph

local atom_module = require('opencog.atom')
local Atom = atom_module.Atom

local Node = setmetatable({}, {__index = Atom})
Node.__index = Node

function Node.new(atom_type, name)
    if not name then
        error("Node must have a name")
    end
    
    local self = Atom.new(atom_type, name)
    setmetatable(self, Node)
    return self
end

function Node:is_node()
    return true
end

function Node:is_link()
    return false
end

function Node:get_hash()
    return self.atom_type .. ":" .. self.name
end

function Node:__tostring()
    return string.format("%s('%s')", self.atom_type, self.name)
end

return Node
