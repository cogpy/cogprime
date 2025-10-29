-- OpenCog Atom Implementation in Pure Lua
-- Base class for all atoms in the AtomSpace

local Atom = {}
Atom.__index = Atom

-- UUID generation (simple implementation)
local function generate_uuid()
    local template = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'
    return string.gsub(template, '[xy]', function(c)
        local v = (c == 'x') and math.random(0, 0xf) or math.random(8, 0xb)
        return string.format('%x', v)
    end)
end

-- TruthValue class
local TruthValue = {}
TruthValue.__index = TruthValue

function TruthValue.new(strength, confidence)
    local self = setmetatable({}, TruthValue)
    self.strength = math.max(0.0, math.min(1.0, strength or 1.0))
    self.confidence = math.max(0.0, math.min(1.0, confidence or 1.0))
    return self
end

function TruthValue:__tostring()
    return string.format("TruthValue(strength=%.3f, confidence=%.3f)", 
                        self.strength, self.confidence)
end

-- AttentionValue class
local AttentionValue = {}
AttentionValue.__index = AttentionValue

function AttentionValue.new(sti, lti, vlti)
    local self = setmetatable({}, AttentionValue)
    self.sti = sti or 0.0    -- Short-Term Importance
    self.lti = lti or 0.0    -- Long-Term Importance
    self.vlti = vlti or false -- Very Long-Term Importance flag
    return self
end

function AttentionValue:__tostring()
    return string.format("AttentionValue(sti=%.3f, lti=%.3f, vlti=%s)", 
                        self.sti, self.lti, tostring(self.vlti))
end

-- Base Atom class
function Atom.new(atom_type, name)
    local self = setmetatable({}, Atom)
    self.atom_type = atom_type
    self.name = name
    self.id = generate_uuid()
    self.tv = TruthValue.new()
    self.av = AttentionValue.new()
    self.incoming_set = {}  -- Links that contain this atom
    self.atomspace = nil    -- Reference to containing atomspace
    self.values = {}        -- Additional key-value pairs
    return self
end

function Atom:is_node()
    return false -- Override in Node class
end

function Atom:is_link()
    return false -- Override in Link class
end

function Atom:get_hash()
    return self.id
end

function Atom:get_value(key)
    return self.values[key]
end

function Atom:set_value(key, value)
    self.values[key] = value
end

function Atom:set_truth_value(tv)
    self.tv = tv
end

function Atom:get_truth_value()
    return self.tv
end

function Atom:set_attention_value(av)
    self.av = av
end

function Atom:get_attention_value()
    return self.av
end

function Atom:add_to_incoming_set(link)
    table.insert(self.incoming_set, link)
end

function Atom:remove_from_incoming_set(link)
    for i, l in ipairs(self.incoming_set) do
        if l.id == link.id then
            table.remove(self.incoming_set, i)
            return
        end
    end
end

function Atom:get_incoming_set()
    return self.incoming_set
end

-- Wildcard for pattern matching
local Wildcard = {}
Wildcard.__index = Wildcard

function Wildcard.new()
    local self = setmetatable({}, Wildcard)
    self._is_wildcard = true
    return self
end

function Wildcard:__tostring()
    return "*"
end

-- Export classes
return {
    Atom = Atom,
    TruthValue = TruthValue,
    AttentionValue = AttentionValue,
    Wildcard = Wildcard
}
