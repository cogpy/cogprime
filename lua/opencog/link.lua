-- OpenCog Link Implementation in Pure Lua
-- Link class representing hyperedges in the AtomSpace hypergraph

local atom_module = require('opencog.atom')
local Atom = atom_module.Atom

local Link = setmetatable({}, {__index = Atom})
Link.__index = Link

function Link.new(atom_type, outgoing_set)
    local self = Atom.new(atom_type)
    setmetatable(self, Link)
    self.outgoing_set = outgoing_set or {}
    return self
end

function Link:is_node()
    return false
end

function Link:is_link()
    return true
end

function Link:get_hash()
    local parts = {self.atom_type}
    for _, atom in ipairs(self.outgoing_set) do
        table.insert(parts, atom.id)
    end
    return table.concat(parts, ":")
end

function Link:get_arity()
    return #self.outgoing_set
end

function Link:get_outgoing_set()
    return self.outgoing_set
end

function Link:__tostring()
    local outgoing_repr = {}
    for _, atom in ipairs(self.outgoing_set) do
        table.insert(outgoing_repr, tostring(atom))
    end
    return string.format("%s(%s)", self.atom_type, table.concat(outgoing_repr, ", "))
end

return Link
