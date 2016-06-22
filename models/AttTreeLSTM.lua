--[[

   Attention-based Child-Sum Tree-LSTM.

--]]

local AttTreeLSTM, parent = torch.class('nn.AttTreeLSTM', 'nn.TreeLSTM')

function AttTreeLSTM:__init(config)
    parent.__init(self, config)
    self.gate_output = config.gate_output
    if self.gate_output == nil then self.gate_output = true end

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

function AttTreeLSTM:new_composer()
    local input = nn.Identity()()
    local child_c = nn.Identity()()
    local child_h = nn.Identity()()
    local atte_h = nn.Identity()()
    local M = nn.Tanh()(nn.CAddTable() {
        nn.Linear(self.mem_dim, self.mem_dim, false)(child_h),
        nn.Linear(self.mem_dim, self.mem_dim, false)(atte_h)
    })
    local temp = nn.Linear(self.mem_dim, 1)(M)
    local attention_weights = (task == true)
            and nn.Transpose({1,2})(nn.SoftMax()(nn.Transpose({1,2})(temp)))
            or nn.Transpose({1,2})(nn.Sigmoid()(nn.Transpose({1,2})(temp)))
    local child_h_att = nn.MM(true, false)({ child_h, attention_weights })
    local child_h_sum = nn.Reshape(self.mem_dim)(child_h_att)

    local i = nn.Sigmoid()(nn.CAddTable() {
        nn.Linear(self.in_dim, self.mem_dim)(input),
        nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
    })
    local f = nn.Sigmoid()(nn.CRowAddTable() {
        nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h),
        nn.Linear(self.in_dim, self.mem_dim)(input),
    })
    local update = nn.Tanh()(nn.CAddTable() {
        nn.Linear(self.in_dim, self.mem_dim)(input),
        nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
    })
    local c = nn.CAddTable() {
        nn.CMulTable() { i, update },
        nn.Sum(1)(nn.CMulTable() { f, child_c })
    }

    local h
    if self.gate_output then
        local o = nn.Sigmoid()(nn.CAddTable() {
            nn.Linear(self.in_dim, self.mem_dim)(input),
            nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
        })
        h = nn.CMulTable() { o, nn.Tanh()(c) }
    else
        h = nn.Tanh()(c)
    end

    local composer = nn.gModule({ input, child_c, child_h, atte_h }, { c, h })
    if self.composer ~= nil then
        share_params(composer, self.composer)
    end
    return composer
end

function AttTreeLSTM:new_output_module()
    if self.output_module_fn == nil then return nil end
    local output_module = self.output_module_fn()
    if self.output_module ~= nil then
        share_params(output_module, self.output_module)
    end
    return output_module
end

function AttTreeLSTM:forward(tree, inputs, attent)
    local loss = 0
    for i = 1, tree.num_children do
        local _, child_loss = self:forward(tree.children[i], inputs, attent)
        loss = loss + child_loss
    end
    local child_c, child_h = self:get_child_states(tree)
    local att_input
    if tree.num_children == 0 then
        att_input = torch.Tensor(1, self.mem_dim)
        att_input[1] = attent
    else
        att_input = torch.Tensor(tree.num_children, self.mem_dim)
        for i = 1, tree.num_children do att_input[i] = attent end
    end
    self:allocate_module(tree, 'composer')
    tree.state = tree.composer:forward { inputs[tree.idx], child_c, child_h, att_input }
    child_h = nil
    child_c = nil
    att_input = nil
    if self.output_module ~= nil then
        self:allocate_module(tree, 'output_module')
        tree.output = tree.output_module:forward(tree.state[2])
        if self.train and tree.gold_label ~= nil then
            loss = loss + self.criterion:forward(tree.output, tree.gold_label)
        end
    end
    return tree.state, loss
end

function AttTreeLSTM:backward(tree, inputs, attent, grad)
    local grad_inputs = torch.Tensor(inputs:size())
    local grad_attent = torch.Tensor(attent:size())
    self:_backward(tree, inputs, attent, grad, grad_inputs, grad_attent)
    return { grad_inputs, grad_attent }
end

function AttTreeLSTM:_backward(tree, inputs, attent, grad, grad_inputs, grad_attent)
    local output_grad = self.mem_zeros
    if tree.output ~= nil and tree.gold_label ~= nil then
        output_grad = tree.output_module:backward(tree.state[2], self.criterion:backward(tree.output, tree.gold_label))
    end
    self:free_module(tree, 'output_module')
    tree.output = nil

    local child_c, child_h = self:get_child_states(tree)
    local att_input
    if tree.num_children == 0 then
        att_input = torch.Tensor(1, self.mem_dim)
        att_input[1] = attent
    else
        att_input = torch.Tensor(tree.num_children, self.mem_dim)
        for i = 1, tree.num_children do att_input[i] = attent end
    end
    local composer_grad = tree.composer:backward({ inputs[tree.idx], child_c, child_h, att_input },
        { grad[1], grad[2] + output_grad })
    att_input = nil
    self:free_module(tree, 'composer')
    tree.state = nil

    grad_inputs[tree.idx] = composer_grad[1]
    grad_attent = composer_grad[4]
    local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]
    composer_grad = nil
    for i = 1, tree.num_children do
        self:_backward(tree.children[i], inputs, attent, { child_c_grads[i], child_h_grads[i] }, grad_inputs, grad_attent)
    end
    child_c_grads = nil
    child_h_grads = nil
end

function AttTreeLSTM:clean(tree)
    self:free_module(tree, 'composer')
    self:free_module(tree, 'output_module')
    tree.state = nil
    tree.output = nil
    for i = 1, tree.num_children do
        self:clean(tree.children[i])
    end
end

function AttTreeLSTM:parameters()
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

function AttTreeLSTM:get_child_states(tree)
    local child_c, child_h
    if tree.num_children == 0 then
        child_c = torch.zeros(1, self.mem_dim)
        child_h = torch.zeros(1, self.mem_dim)
    else
        child_c = torch.Tensor(tree.num_children, self.mem_dim)
        child_h = torch.Tensor(tree.num_children, self.mem_dim)
        for i = 1, tree.num_children do
            child_c[i], child_h[i] = unpack(tree.children[i].state)
        end
    end
    return child_c, child_h
end