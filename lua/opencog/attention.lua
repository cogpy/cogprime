-- OpenCog Attention Allocation in Pure Lua
-- Economic Attention Network (ECAN) implementation

local AttentionBank = {}
AttentionBank.__index = AttentionBank

-- Attention configuration constants
local STI_TO_LTI_CONVERSION_RATE = 0.1

function AttentionBank.new(atomspace, config)
    local self = setmetatable({}, AttentionBank)
    self.atomspace = atomspace
    
    -- Configuration
    config = config or {}
    self.max_sti = config.max_sti or 100.0
    self.min_sti = config.min_sti or -100.0
    self.sti_threshold = config.sti_threshold or 0.0
    self.lti_threshold = config.lti_threshold or 0.0
    self.decay_rate = config.decay_rate or 0.95
    self.importance_diffusion_rate = config.importance_diffusion_rate or 0.1
    
    -- Total STI in the system
    self.total_sti = config.total_sti or 1000.0
    
    return self
end

-- Update attention values based on importance spreading
function AttentionBank:update_attention()
    local atoms = self.atomspace:get_all_atoms()
    
    -- Spread importance along links
    for _, atom in ipairs(atoms) do
        if atom:is_link() then
            self:spread_importance(atom)
        end
    end
    
    -- Apply decay to STI
    for _, atom in ipairs(atoms) do
        local av = atom:get_attention_value()
        av.sti = av.sti * self.decay_rate
        
        -- Clamp STI values
        av.sti = math.max(self.min_sti, math.min(self.max_sti, av.sti))
        
        atom:set_attention_value(av)
    end
    
    -- Normalize STI to maintain total
    self:normalize_sti()
end

-- Spread importance from a link to its outgoing set
function AttentionBank:spread_importance(link)
    local av = link:get_attention_value()
    local diffusion_amount = av.sti * self.importance_diffusion_rate
    local per_atom = diffusion_amount / #link.outgoing_set
    
    for _, atom in ipairs(link.outgoing_set) do
        local target_av = atom:get_attention_value()
        target_av.sti = target_av.sti + per_atom
        atom:set_attention_value(target_av)
    end
    
    -- Reduce link's STI by the diffused amount
    av.sti = av.sti - diffusion_amount
    link:set_attention_value(av)
end

-- Normalize STI values to maintain total STI in system
function AttentionBank:normalize_sti()
    local atoms = self.atomspace:get_all_atoms()
    local current_total = 0.0
    
    -- Calculate current total STI
    for _, atom in ipairs(atoms) do
        current_total = current_total + atom:get_attention_value().sti
    end
    
    -- Normalize if needed
    if current_total > 0 then
        local scale_factor = self.total_sti / current_total
        
        for _, atom in ipairs(atoms) do
            local av = atom:get_attention_value()
            av.sti = av.sti * scale_factor
            atom:set_attention_value(av)
        end
    end
end

-- Get atoms in attentional focus (STI above threshold)
function AttentionBank:get_attentional_focus()
    local focused = {}
    local atoms = self.atomspace:get_all_atoms()
    
    for _, atom in ipairs(atoms) do
        if atom:get_attention_value().sti >= self.sti_threshold then
            table.insert(focused, atom)
        end
    end
    
    -- Sort by STI (descending)
    table.sort(focused, function(a, b)
        return a:get_attention_value().sti > b:get_attention_value().sti
    end)
    
    return focused
end

-- Stimulate an atom (increase its STI)
function AttentionBank:stimulate(atom, amount)
    local av = atom:get_attention_value()
    av.sti = math.min(self.max_sti, av.sti + amount)
    atom:set_attention_value(av)
end

-- Get top N atoms by STI
function AttentionBank:get_top_atoms(n)
    local atoms = self.atomspace:get_all_atoms()
    
    -- Sort by STI (descending)
    table.sort(atoms, function(a, b)
        return a:get_attention_value().sti > b:get_attention_value().sti
    end)
    
    local top = {}
    for i = 1, math.min(n, #atoms) do
        table.insert(top, atoms[i])
    end
    
    return top
end

-- Convert STI to LTI for important atoms
function AttentionBank:consolidate_attention()
    local atoms = self.atomspace:get_all_atoms()
    
    for _, atom in ipairs(atoms) do
        local av = atom:get_attention_value()
        
        -- Convert some STI to LTI for atoms with high STI
        if av.sti > self.sti_threshold then
            local conversion_amount = av.sti * STI_TO_LTI_CONVERSION_RATE
            av.sti = av.sti - conversion_amount
            av.lti = av.lti + conversion_amount
            
            atom:set_attention_value(av)
        end
    end
end

-- Forget atoms with low importance
function AttentionBank:forget_low_importance(threshold)
    threshold = threshold or self.lti_threshold
    local atoms = self.atomspace:get_all_atoms()
    local to_remove = {}
    
    for _, atom in ipairs(atoms) do
        local av = atom:get_attention_value()
        if av.lti < threshold and av.sti < self.sti_threshold then
            table.insert(to_remove, atom)
        end
    end
    
    -- Remove low importance atoms
    for _, atom in ipairs(to_remove) do
        self.atomspace:remove(atom)
    end
    
    return #to_remove
end

return AttentionBank
