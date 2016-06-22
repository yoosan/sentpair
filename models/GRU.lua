--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 16/6/3, 2015.
 - Licence MIT
--]]

--[[

 Gated Recurrent Unit. (http://arxiv.org/pdf/1412.3555v1.pdf)

--]]

local GRU, parent = torch.class('nn.GRU', 'nn.Module')


function GRU:__init(config)
    parent.__init(self)

    self.in_dim = config.in_dim or 300
    self.mem_dim = config.mem_dim or 150
    self.num_layers = config.num_layers or 1

    self.master_cell = self:new_cell()
    self.depth = 0
    self.cells = {}

    local htable_init, htable_grad
    if self.num_layers == 1 then
        htable_init = torch.zeros(self.mem_dim)
        htable_grad = torch.zeros(self.mem_dim)
    else
        htable_init, htable_grad = {}, {}
        for i = 1, self.num_layers do
            htable_init[i] = torch.zeros(self.mem_dim)
            htable_grad[i] = torch.zeros(self.mem_dim)
        end
    end
    self.initial_values = htable_init
    self.gradInput = {
        torch.zeros(self.in_dim),
        htable_grad
    }
    self.params, self.grad_params = self.master_cell:getParameters()
end

-- Instantiate a new GRU cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function GRU:new_cell()
    local input = nn.Identity()()
    local htable_p = nn.Identity()()

    -- output = W^{(x)}xv + U^{(h)}hv + b
    local function new_gate(indim, xv, hv)
        local i2h = nn.Linear(indim, self.mem_dim)(xv)
        local h2h = nn.Linear(self.mem_dim, self.mem_dim)(hv)
        return nn.CAddTable()({ i2h, h2h })
    end

    -- multilayer GRU
    local htable = {}
    for layer = 1, self.num_layers do
        local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)

        -- update and reset gates
        local z = (layer == 1) and nn.Sigmoid()(new_gate(self.in_dim, input, h_p))
                or nn.Sigmoid()(new_gate(self.mem_dim, htable[layer - 1], h_p))
        local r = (layer == 1) and nn.Sigmoid()(new_gate(self.in_dim, input, h_p))
                or nn.Sigmoid()(new_gate(self.mem_dim, htable[layer - 1], h_p))

        local g = nn.CMulTable()({ r, h_p })
        local h_c = (layer == 1) and nn.Tanh()(new_gate(self.in_dim, input, g))
                or nn.Tanh()(new_gate(self.mem_dim, htable[layer - 1], g))

        local z_d = nn.AddConstant(1, false)(nn.MulConstant(-1, false)(z))
        local p1 = nn.CMulTable()({ z_d, h_c })
        local p2 = nn.CMulTable()({ z, h_p })

        local next_h = nn.CAddTable()({ p1, p2 })

        htable[layer] = next_h
    end

    -- if GRU is single-layered, this makes htable/ctable Tensors (instead of tables).
    -- this avoids some quirks with nngraph involving tables of size 1.
    --    htable = nn.Identity()(htable)
    local cell = nn.gModule({ input, htable_p }, htable)

    -- share parameters
    if self.master_cell then
        share_params(cell, self.master_cell)
    end
    return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional GRUS).
-- Returns the final hidden state of the GRU.
function GRU:forward(inputs, reverse)
    local size = inputs:size(1)
    for t = 1, size do
        local input = reverse and inputs[size - t + 1] or inputs[t]
        self.depth = self.depth + 1
        local cell = self.cells[self.depth]
        if cell == nil then
            cell = self:new_cell()
            self.cells[self.depth] = cell
        end
        local prev_output
        if self.depth > 1 then
            prev_output = self.cells[self.depth - 1].output
        else
            prev_output = self.initial_values
        end

        local outputs = cell:forward({ input, prev_output })
        local htable = outputs
        if self.num_layers == 1 then
            self.output = htable
        else
            self.output = {}
            for i = 1, self.num_layers do
                self.output[i] = htable[i]
            end
        end
    end
    return self.output
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function GRU:backward(inputs, grad_outputs, reverse)
    local size = inputs:size(1)
    if self.depth == 0 then
        error("No cells to backpropagate through")
    end

    local input_grads = torch.Tensor(inputs:size())
    for t = size, 1, -1 do
        local input = reverse and inputs[size - t + 1] or inputs[t]
        local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
        local cell = self.cells[self.depth]
        local grads = self.gradInput[2]
        if self.num_layers == 1 then
            grads:add(grad_output)
        else
            for i = 1, self.num_layers do
                grads[i]:add(grad_output[i])
            end
        end

        local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                or self.initial_values
        self.gradInput = cell:backward({ input, prev_output }, grads)
        if reverse then
            input_grads[size - t + 1] = self.gradInput[1]
        else
            input_grads[t] = self.gradInput[1]
        end
        self.depth = self.depth - 1
    end
    self:forget() -- important to clear out state
    return input_grads
end

function GRU:share(gru, ...)
    if self.in_dim ~= gru.in_dim then error("GRU input dimension mismatch") end
    if self.mem_dim ~= gru.mem_dim then error("GRU memory dimension mismatch") end
    if self.num_layers ~= gru.num_layers then error("GRU layer count mismatch") end
    share_params(self.master_cell, gru.master_cell, ...)
end

function GRU:zeroGradParameters()
    self.master_cell:zeroGradParameters()
end

function GRU:parameters()
    return self.master_cell:parameters()
end

-- Clear saved gradients
function GRU:forget()
    self.depth = 0
    for i = 1, #self.gradInput do
        local gradInput = self.gradInput[i]
        if type(gradInput) == 'table' then
            for _, t in pairs(gradInput) do t:zero() end
        else
            self.gradInput[i]:zero()
        end
    end
end