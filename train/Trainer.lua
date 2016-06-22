--[[

  Semantic relatedness prediction using LSTMs.

--]]

local Trainer = torch.class('Trainer')

function Trainer:__init(config)
    self.task           = config.task           or 'SICK'
    self.mem_dim        = config.mem_dim        or 150
    self.learning_rate  = config.learning_rate  or 0.05
    self.batch_size     = config.batch_size     or 25
    self.num_layers     = config.num_layers     or 1
    self.reg            = config.reg            or 1e-4
    self.structure      = config.structure      or 'lstm' -- {lstm, bilstm}
    self.feats_dim      = config.feats_dim      or 50

    -- word embedding
    self.emb_vecs = config.emb_vecs
    self.emb_dim = config.emb_vecs:size(2)

    -- optimizer config
    self.optim_func = config.optim_func or optim.adagrad
    self.optim_state = {
        learningRate = self.learning_rate,
    }

    -- number of classes and criterion
    if self.task == 'MSRP' or self.task == 'WQA' then
        self.num_classes = 2
        self.criterion = nn.BCECriterion()
    elseif self.task == 'SNLI' then
        self.num_classes = 3
        self.criterion = nn.ClassNLLCriterion()
    elseif self.task == 'SICK' then
        self.num_classes = 5
        self.criterion = nn.DistKLDivCriterion()
    else
        error('No such task! The tasks are SICK, SNLI, MSRP and WQA')
    end

    -- initialize model
    local model_config = {
        in_dim = self.emb_dim,
        mem_dim = self.mem_dim,
        num_layers = self.num_layers,
        gate_output = false,
    }

    if self.structure == 'lstm' then
        self.lmodel = nn.LSTM(model_config)
        self.rmodel = nn.LSTM(model_config)
    elseif self.structure == 'gru' then
        self.lmodel = nn.GRU(model_config)
        self.rmodel = nn.GRU(model_config)
    elseif self.structure == 'treelstm' then
        self.model = nn.ChildSumTreeLSTM(model_config)
    elseif self.structure == 'treegru' then
        self.model = nn.ChildSumTreeLSTM(model_config)
    elseif self.structure == 'atreelstm' then
        self.model = nn.AttTreeLSTM(model_config)
        self.latt = nn.LSTM(model_config)
        self.ratt = nn.LSTM(model_config)
    elseif self.structure == 'atreegru' then
        self.model = nn.AttTreeGRU(model_config)
        self.latt = nn.GRU(model_config)
        self.ratt = nn.GRU(model_config)
    else
        error('invalid model type: ' .. self.structure)
    end

    -- output module
    self.output_module = self:new_output_module()
    local modules = nn.Parallel()
    if self.structure == 'lstm' or self.structure == 'gru' then
        modules:add(self.lmodel)
    elseif self.structure == 'treelstm' or self.structure == 'treegru' then
        modules:add(self.model)
    else
        modules:add(self.model):add(self.latt)
    end
    modules:add(self.output_module)
    self.params, self.grad_params = modules:getParameters()

    -- share must only be called after getParameters, since this changes the
    -- location of the parameters
    if self.structure == 'lstm' or self.structure == 'gru' then
        share_params(self.rmodel, self.lmodel)
    elseif self.structure == 'atreelstm' or self.structure == 'atreegru' then
        share_params(self.ratt, self.latt)
    end
end

function Trainer:new_output_module()
    local lvec, rvec, inputs, input_dim
    if self.structure == 'lstm' or self.structure == 'gru' then
        -- standard (left-to-right) LSTM
        input_dim = 2 * self.num_layers * self.mem_dim
        local linput, rinput = nn.Identity()(), nn.Identity()()
        if self.num_layers == 1 then
            lvec, rvec = linput, rinput
        else
            lvec, rvec = nn.JoinTable(1)(linput), nn.JoinTable(1)(rinput)
        end
        inputs = { linput, rinput }
    else
        input_dim = 2 * self.mem_dim
        local linput, rinput = nn.Identity()(), nn.Identity()()
        lvec, rvec = linput, rinput
        inputs = { linput, rinput }
    end
    local mult_dist = nn.CMulTable() { lvec, rvec }
    local add_dist = nn.Abs()(nn.CSubTable() { lvec, rvec })
    local vec_dist_feats = nn.JoinTable(1) { mult_dist, add_dist }
    local vecs_to_input = nn.gModule(inputs, { vec_dist_feats })

    -- define similarity model architecture
    local classifier
    if self.task == 'MSRP' or self.task == 'WQA' then
        classifier = nn.Sigmoid()
    elseif self.task == 'SICK' or self.task == 'SNLI' then
        classifier = nn.LogSoftMax()
    end
    local output_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(input_dim, self.feats_dim))
    :add(nn.Sigmoid())
    :add(nn.Linear(self.feats_dim, self.num_classes))
    :add(classifier)
    return output_module
end

function Trainer:train(dataset)
    if self.structure == 'lstm' or self.structure == 'gru' then
        self.lmodel:training()
        self.rmodel:training()
    elseif self.structure == 'treelstm' or self.structure == 'treegru' then
        self.model:training()
    else
        self.model:training()
        self.latt:training()
        self.ratt:training()
    end

    local indices = torch.randperm(dataset.size)
    local zeros = torch.zeros(self.mem_dim)
    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        -- get target distributions for batch
        local targets = torch.zeros(batch_size, self.num_classes)
        if self.task == 'SICK' then
            for j = 1, batch_size do
                local sim = dataset.labels[indices[i + j - 1]] * (self.num_classes - 1) + 1
                local ceil, floor = math.ceil(sim), math.floor(sim)
                if ceil == floor then
                    targets[{j, floor}] = 1
                else
                    targets[{j, floor}] = ceil - sim
                    targets[{j, ceil}] = sim - floor
                end
            end
        else
            for j = 1, batch_size do
                local label = dataset.labels[indices[i + j - 1]]
                targets[{j, label}] = 1
            end
        end
        local feval = function(x)
            self.grad_params:zero()
            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]
                local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
                local ltree, rtree = dataset.ltrees[idx], dataset.rtrees[idx]
                local linputs = self.emb_vecs:index(1, lsent:long()):double()
                local rinputs = self.emb_vecs:index(1, rsent:long()):double()

                -- get sentence representations
                local inputs, l_seqrep, r_seqrep
                if self.structure == 'lstm' or self.structure == 'gru' then
                    inputs = {
                        self.lmodel:forward(linputs),
                        self.rmodel:forward(rinputs)
                    }
                elseif self.structure == 'treelstm' then
                    inputs = {
                        self.model:forward(ltree, linputs)[2],
                        self.model:forward(rtree, rinputs)[2]
                    }
                elseif self.structure == 'treegru' then
                    inputs = {
                        self.model:forward(ltree, linputs),
                        self.model:forward(rtree, rinputs)
                    }
                elseif self.structure == 'atreelstm' then
                    l_seqrep = self.latt:forward(linputs)
                    r_seqrep = self.ratt:forward(rinputs)
                    inputs = {
                        self.model:forward(ltree, linputs, l_seqrep)[2],
                        self.model:forward(rtree, rinputs, r_seqrep)[2]
                    }
                else
                    l_seqrep = self.latt:forward(linputs)
                    r_seqrep = self.ratt:forward(rinputs)
                    inputs = {
                        self.model:forward(ltree, linputs, l_seqrep),
                        self.model:forward(rtree, rinputs, r_seqrep)
                    }
                end
                -- compute relatedness
                local output = self.output_module:forward(inputs)

                -- compute loss and backpropagate
                local example_loss = self.criterion:forward(output, targets[j])
                loss = loss + example_loss
                local out_grad = self.criterion:backward(output, targets[j])
                local rep_grad = self.output_module:backward(inputs, out_grad)
                if self.structure == 'lstm' or self.structure == 'gru' then
                    self:RNN_backward(lsent, rsent, linputs, rinputs, rep_grad)
                elseif self.sturcture == 'treegru' then
                    self.model:backward(ltree, linputs, rep_grad[1])
                    self.model:backward(rtree, rinputs, rep_grad[2])
                elseif self.structure == 'treelstm' then
                    self.model:backward(ltree, linputs, {zeros, rep_grad[1]})
                    self.model:backward(rtree, rinputs, {zeros, rep_grad[2]})
                elseif self.structure == 'atreegru' then
                    self.model:backward(ltree, linputs, l_seqrep, rep_grad[1])
                    self.model:backward(rtree, rinputs, r_seqrep, rep_grad[2])
                elseif self.structure == 'atreelstm' then
                    self.model:backward(ltree, linputs, l_seqrep, {zeros, rep_grad[1]})
                    self.model:backward(rtree, rinputs, r_seqrep, {zeros, rep_grad[2]})
                end
            end

            loss = loss / batch_size
            self.grad_params:div(batch_size)

            -- regularization
            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            return loss, self.grad_params
        end

        optim.adagrad(feval, self.params, self.optim_state)
    end
    xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function Trainer:RNN_backward(lsent, rsent, linputs, rinputs, rep_grad)
    local lgrad, rgrad
    if self.num_layers == 1 then
        lgrad = torch.zeros(lsent:nElement(), self.mem_dim)
        rgrad = torch.zeros(rsent:nElement(), self.mem_dim)
        lgrad[lsent:nElement()] = rep_grad[1]
        rgrad[rsent:nElement()] = rep_grad[2]
    else
        lgrad = torch.zeros(lsent:nElement(), self.num_layers, self.mem_dim)
        rgrad = torch.zeros(rsent:nElement(), self.num_layers, self.mem_dim)
        for l = 1, self.num_layers do
            lgrad[{ lsent:nElement(), l, {} }] = rep_grad[1][l]
            rgrad[{ rsent:nElement(), l, {} }] = rep_grad[2][l]
        end
    end
    self.lmodel:backward(linputs, lgrad)
    self.rmodel:backward(rinputs, rgrad)
end

-- Predict the similarity of a sentence pair.
function Trainer:predict(lsent, rsent, ltree, rtree)
    if self.structure == 'lstm' or self.structure == 'gru' then
        self.lmodel:evaluate()
        self.rmodel:evaluate()
    elseif self.structure == 'treelstm' or self.structure == 'treegru' then
        self.model:evaluate()
    else
        self.model:evaluate()
        self.latt:evaluate()
        self.ratt:evaluate()
    end
    local linputs = self.emb_vecs:index(1, lsent:long()):double()
    local rinputs = self.emb_vecs:index(1, rsent:long()):double()
    local inputs
    local inputs, l_seqrep, r_seqrep
    if self.structure == 'lstm' or self.structure == 'gru' then
        inputs = {
            self.lmodel:forward(linputs),
            self.rmodel:forward(rinputs)
        }
    elseif self.structure == 'treelstm' then
        inputs = {
            self.model:forward(ltree, linputs)[2],
            self.model:forward(rtree, rinputs)[2]
        }
    elseif self.structure == 'treegru' then
        inputs = {
            self.model:forward(ltree, linputs),
            self.model:forward(rtree, rinputs)
        }
    else
        l_seqrep = self.latt:forward(linputs)
        r_seqrep = self.ratt:forward(rinputs)
        inputs = {
            self.model:forward(ltree, linputs, l_seqrep),
            self.model:forward(rtree, rinputs, r_seqrep)
        }
    end
    local output = self.output_module:forward(inputs)
    if self.structure == 'lstm' or self.structure == 'gru' then
        self.lmodel:forget()
        self.rmodel:forget()
    elseif self.structure == 'treelstm' or self.structure == 'treegru' then
        self.model:clean(ltree)
        self.model:clean(rtree)
    else
        self.model:clean(ltree)
        self.model:clean(rtree)
        self.latt:forget()
        self.ratt:forget()
    end
    if self.task == 'SICK' then
        return torch.range(1, 5):dot(output:exp())
    else
        return stats.argmax(output)
    end
end

-- Produce similarity predictions for each sentence pair in the dataset.
function Trainer:eval(dataset)
    local predictions = torch.Tensor(dataset.size)
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
        if self.structure == 'lstm' or self.structure == 'gru' then
            predictions[i] = self:predict(lsent, rsent)
        else
            local ltree, rtree = dataset.ltrees[i], dataset.rtrees[i]
            predictions[i] = self:predict(lsent, rsent, ltree, rtree)
        end
    end
    return predictions
end

function Trainer:print_config()
    local num_params = self.params:nElement()
    local num_sim_params = self:new_output_module():getParameters():nElement()
    printf('%-25s = %d\n', 'num params', num_params)
    printf('%-25s = %d\n', 'num compositional params', num_params - num_sim_params)
    printf('%-25s = %d\n', 'word vector dim', self.emb_dim)
    printf('%-25s = %d\n', 'LSTM memory dim', self.mem_dim)
    printf('%-25s = %.2e\n', 'regularization strength', self.reg)
    printf('%-25s = %d\n', 'minibatch size', self.batch_size)
    printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
    printf('%-25s = %s\n', 'LSTM structure', self.structure)
    printf('%-25s = %d\n', 'LSTM layers', self.num_layers)
    printf('%-25s = %d\n', 'sim module hidden dim', self.feats_dim)
end

-- Serialization
function Trainer:save(path)
    local config = {
        batch_size = self.batch_size,
        emb_vecs = self.emb_vecs:float(),
        learning_rate = self.learning_rate,
        num_layers = self.num_layers,
        mem_dim = self.mem_dim,
        sim_nhidden = self.sim_nhidden,
        reg = self.reg,
        structure = self.structure,
    }

    torch.save(path, {
        params = self.params,
        config = config,
    })
end
