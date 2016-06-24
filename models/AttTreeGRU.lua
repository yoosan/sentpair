--[[

   Attention-based Child-Sum Tree-GRU.

--]]

local AttTreeGRU, parent = torch.class('nn.AttTreeGRU', 'nn.TreeGRU')

function AttTreeGRU:__init(config)
    parent.__init(self, config)

    -- a function that instantiates an output module that takes the hidden state h as input
    self.output_module_fn = config.output_module_fn
    self.criterion = config.criterion

    -- composition module
    self.composer = self:new_composer()
    self.composers = {}

    -- output module
    self.output_module = self:new_output_module()
    self.output_modules = {}

    -- task
    self.task = config.task or true
end

function AttTreeGRU:new_composer()
    local input = nn.Identity()()
    local child_h = nn.Identity()()
    local prev_res = nn.Identity()()
    local M = nn.Tanh()(nn.CAddTable() {
        nn.Linear(self.mem_dim, self.mem_dim, false)(child_h),
        nn.Linear(self.mem_dim, self.mem_dim, false)(prev_res)
    })
    local temp = nn.Linear(self.mem_dim, 1)(M)
    local attention_weights = (task == true)
            and nn.Transpose({1,2})(nn.SoftMax()(nn.Transpose({1,2})(temp)))
            or nn.Transpose({1,2})(nn.Sigmoid()(nn.Transpose({1,2})(temp)))
    local child_h_att = nn.MM(true, false)({ child_h, attention_weights })
    local child_h_sum = nn.Reshape(self.mem_dim)(child_h_att)

    local z = nn.Sigmoid()(nn.CAddTable() {
        nn.Linear(self.in_dim, self.mem_dim)(input),
        nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
    })
    local r = nn.Sigmoid()(attrnn.CRowAddTable() {
        nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h),
        nn.Linear(self.in_dim, self.mem_dim)(input),
    })
    local g = nn.Sum(1)(nn.CMulTable() { r, child_h })

    local h_candidate = nn.Tanh()(nn.CAddTable() {
        nn.Linear(self.in_dim, self.mem_dim)(input),
        nn.Linear(self.mem_dim, self.mem_dim)(g)
    })
    local z_d = nn.AddConstant(1, false)(nn.MulConstant(-1, false)(z))

    local p1 = nn.CMulTable()({ z, h_candidate })
    local p2 = nn.CMulTable()({ z_d, child_h_sum })

    local h_next = nn.CAddTable()({ p1, p2 })

    local composer = nn.gModule({ input, child_h, prev_res }, { h_next })
    if self.composer ~= nil then
        share_params(composer, self.composer)
    end
    return composer
end

function AttTreeGRU:new_output_module()
    if self.output_module_fn == nil then return nil end
    local output_module = self.output_module_fn()
    if self.output_module ~= nil then
        share_params(output_module, self.output_module)
    end
    return output_module
end

function AttTreeGRU:forward(tree, inputs, seq_rep)
    for i = 1, tree.num_children do
        self:forward(tree.children[i], inputs, seq_rep)
    end
    local child_h = self:get_child_states(tree)
    local seq_rep_feed
    if tree.num_children == 0 then
        seq_rep_feed = torch.Tensor(1, self.mem_dim)
        seq_rep_feed[1] = seq_rep
    else
        seq_rep_feed = torch.Tensor(tree.num_children, self.mem_dim)
        for i = 1, tree.num_children do seq_rep_feed[i] = seq_rep end
    end
    self:allocate_module(tree, 'composer')
    tree.state = tree.composer:forward { inputs[tree.idx], child_h, seq_rep_feed }
    seq_rep_feed = nil
    child_h = nil
    return tree.state
end

function AttTreeGRU:backward(tree, inputs, seq_rep, grad)
    local grad_inputs = torch.Tensor(inputs:size())
    local grad_rep = torch.Tensor(seq_rep:size())
    self:_backward(tree, inputs, seq_rep, grad, grad_inputs, grad_rep)
    return { grad_inputs, grad_rep }
end

function AttTreeGRU:_backward(tree, inputs, seq_rep, grad, grad_inputs, grad_rep)
    local output_grad = self.mem_zeros
    if tree.output ~= nil and tree.gold_label ~= nil then
        output_grad = tree.output_module:backward(tree.state, self.criterion:backward(tree.output, tree.gold_label))
    end
    self:free_module(tree, 'output_module')
    tree.output = nil

    local child_h = self:get_child_states(tree)
    local seq_rep_feed
    if tree.num_children == 0 then
        seq_rep_feed = torch.Tensor(1, self.mem_dim)
        seq_rep_feed[1] = seq_rep
    else
        seq_rep_feed = torch.Tensor(tree.num_children, self.mem_dim)
        for i = 1, tree.num_children do seq_rep_feed[i] = seq_rep end
    end
    local composer_grad = tree.composer:backward({ inputs[tree.idx], child_h, seq_rep_feed },
        grad + output_grad)
    child_h = nil
    seq_rep_feed = nil
    self:free_module(tree, 'composer')
    tree.state = nil

    grad_inputs[tree.idx] = composer_grad[1]
    local child_h_grads = composer_grad[2]
    grad_rep = composer_grad[3][1]
    composer_grad = nil

    for i = 1, tree.num_children do
        self:_backward(tree.children[i], inputs, seq_rep, child_h_grads[i], grad_inputs, grad_rep)
    end
end

function AttTreeGRU:clean(tree)
    self:free_module(tree, 'composer')
    self:free_module(tree, 'output_module')
    tree.state = nil
    tree.output = nil
    for i = 1, tree.num_children do
        self:clean(tree.children[i])
    end
end

function AttTreeGRU:parameters()
    local params, grad_params = {}, {}
    local cp, cg = self.composer:parameters()
    tablex.insertvalues(params, cp)
    tablex.insertvalues(grad_params, cg)
    if self.output_module ~= nil then
        local op, og = self.output_module:parameters()
        tablex.insertvalues(params, op)
        tablex.insertvalues(grad_params, og)
    end
    return params, grad_params
end

function AttTreeGRU:get_child_states(tree)
    local child_h
    if tree.num_children == 0 then
        child_h = torch.zeros(1, self.mem_dim)
    else
        child_h = torch.Tensor(tree.num_children, self.mem_dim)
        for i = 1, tree.num_children do
            child_h[i] = tree.children[i].state
        end
    end
    return child_h
end
