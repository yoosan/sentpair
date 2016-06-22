--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 10/21/15, 2015.
 - Licence MIT
--]]

local Tester = torch.class('Tester')

function Tester:__init(config)
    self.task          = config.task          or 'SICK' -- SICK, SNLI, MSRP, WQA
    self.mem_dim       = config.mem_dim       or 150
    self.learning_rate = config.lr            or 0.05
    self.batch_size    = config.batch_size    or 25
    self.reg           = config.reg           or 1e-4
    self.structure     = config.structure     or 'lstm' -- gru, treelstm, treegru, atreelstm, atreegru
    self.feats_dim     = config.feats_dim     or 50

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

    -- model for training
    local model_config = {
        in_dim = self.emb_dim,
        mem_dim = self.mem_dim,
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
        share_params(self.ratt, self.latt)
    elseif self.structure == 'atreegru' then
        self.model = nn.AttTreeGRU(model_config)
        self.latt = nn.GRU(model_config)
        self.ratt = nn.GRU(model_config)
        share_params(self.ratt, self.latt)
    else
        error('invalid model type: ' .. self.structure)
    end

    -- output module
    self.output_module = self:new_output_module()

    -- parameters for training
    self.modules = nn.Parallel()
    if self.structure == 'lstm' or self.structure == 'gru' then
        self.modules:add(self.lmodel)
    elseif self.structure == 'treelstm' or self.structure == 'treegru' then
        self.modules:add(self.model)
    else
        self.modules:add(self.model):add(self.latt)
    end
    self.modules:add(self.output_module)
    self.params, self.grad_params = self.modules:getParameters()
    share_params(self.rmodel, self.lmodel)
end

function Tester:new_output_module()
    local lrep = nn.Identity()()
    local rrep = nn.Identity()()
    local mul_dist = nn.CMulTable(){lrep, rrep}
    local sub_dist = nn.Abs()(nn.CSubTable(){lrep, rrep})
    local rep_dist_feats = nn.JoinTable(1){mul_dist, sub_dist}
    local feats = nn.gModule({lrep, rrep}, {rep_dist_feats})

    -- output module, feed feats to the classifier
    local classifier
    if self.task == 'MSRP' or self.task == 'WQA' then
        classifier = nn.Sigmoid()
    elseif self.task == 'SICK' or self.task == 'SNLI' then
        classifier = nn.LogSoftMax()
    end
    local out_module = nn.Sequential()
    :add(feats)
    :add(nn.Linear(2 * self.mem_dim, self.feats_dim))
    :add(nn.Sigmoid())
    :add(nn.Linear(self.feats_dim, self.num_classes))
    :add(classifier)
    return out_module
end

function Tester:train(dataset)
    self.lmodel:training()
    self.rmodel:training()

    -- shuffle the dataset
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

                -- forward and get representations of the sent pair
                local lrep, rrep, seq_lrep, seq_rrep
                if self.structure == 'lstm' or self.structure == 'gru' then
                    lrep = self.lmodel:forward(linputs)
                    rrep = self.rmodel:forward(rinputs)
                elseif self.sturcture == 'treelstm' or self.sturcture == 'treegru' then
                    lrep = self.model:forward(ltree, linputs)
                    rrep = self.model:forward(rtree, rinputs)
                else
                    seq_lrep = self.latt:forward(linputs)
                    seq_rrep = self.ratt:forward(rinputs)
                    lrep = self.model:forward(ltree, linputs, seq_lrep)
                    rrep = self.model:forward(rtree, rinputs, seq_rrep)
                end

                -- feed to the output module and compute loss
                local output = self.output_module:forward({lrep, rrep})
                local example_loss = self.criterion:forward(output, targets[j])
                loss = loss + example_loss

                -- backward
                local out_grad = self.criterion:backward(output, targets[j])
                local rep_grads = self.output_module:backward({lrep, rrep}, out_grad)
                local lrep_grad, rrep_grad = rep_grads[1], rep_grads[2]

                -- backward grads of representation
                if self.structure == 'lstm' or self.structure == 'gru' then
                    self:RNN_backward(lsent, rsent, linputs, rinputs, rep_grads)
                elseif self.sturcture == 'treegru' then
                    self.model:backward(ltree, linputs, lrep_grad)
                    self.model:backward(rtree, rinputs, rrep_grad)
                elseif self.structure == 'treelstm' then
                    self.model:backward(ltree, linputs, {zeros, lrep_grad})
                    self.model:backward(rtree, rinputs, {zeros, rrep_grad})
                elseif self.structure == 'atreegru' then
                    self.model:backward(ltree, linputs, seq_lrep, lrep_grad)
                    self.model:backward(rtree, rinputs, seq_rrep, rrep_grad)
                elseif self.structure == 'atreelstm' then
                    self.model:backward(ltree, linputs, seq_lrep, {zeros, lrep_grad})
                    self.model:backward(rtree, rinputs, seq_rrep, {zeros, rrep_grad})
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

function Tester:RNN_backward(lsent, rsent, linputs, rinputs, rep_grad)
    local lgrad, rgrad
    lgrad = torch.zeros(lsent:nElement(), self.mem_dim)
    rgrad = torch.zeros(rsent:nElement(), self.mem_dim)
    lgrad[lsent:nElement()] = rep_grad[1]
    rgrad[rsent:nElement()] = rep_grad[2]
    self.lmodel:backward(linputs, lgrad)
    self.rmodel:backward(rinputs, rgrad)
end

function Tester:predict(lsent, rsent, ltree, rtree)
    self.lmodel:evaluate()
    self.rmodel:evaluate()
    local linputs = self.emb_vecs:index(1, lsent:long()):double()
    local rinputs = self.emb_vecs:index(1, rsent:long()):double()

    local inputs
    if self.structure == 'lstm' or self.structure == 'gru' then
        inputs = {
            self.lmodel:forward(linputs),
            self.rmodel:forward(rinputs)
        }
    elseif self.sturcture == 'treelstm' or self.sturcture == 'treegru' then
        inputs = {
            self.model:forward(ltree, linputs),
            self.model:forward(rtree, rinputs)
        }
        self.model:clean(ltree)
        self.model:clean(rtree)
    else
        local seq_lrep = self.latt:forward(linputs)
        local seq_rrep = self.ratt:forward(rinputs)
        inputs = {
            self.model:forward(ltree, linputs, seq_lrep),
            self.model:forward(rtree, rinputs, seq_rrep)
        }
        self.model:clean(ltree)
        self.model:clean(rtree)
        self.latt:forget()
        self.ratt:forget()
    end
    local output = self.output_module:forward(inputs)
    self.lmodel:forget()
    self.rmodel:forget()
    if self.task == 'SICK' then
        return torch.range(1, 5):dot(output:exp())
    else
        return stats.argmax(output)
    end
end

function Tester:eval(dataset)
    local predictions = torch.zeros(dataset.size)
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
        predictions[i] = self:predict(lsent, rsent)
    end
    return predictions
end

function Tester:print_config()
    local num_params = self.params:nElement()
    local num_output_params = self.output_module:getParameters():nElement()
    printf('%-25s = %s\n',   'running task', self.task)
    printf('%-25s = %d\n',   'num params', num_params)
    printf('%-25s = %d\n',   'num compositional params', num_params - num_output_params)
    printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
    printf('%-25s = %d\n',   'memory dim', self.mem_dim)
    printf('%-25s = %d\n',   'feats dim', self.feats_dim)
    printf('%-25s = %.2e\n', 'regularization strength', self.reg)
    printf('%-25s = %d\n',   'minibatch size', self.batch_size)
    printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
    printf('%-25s = %s\n',   'model structure', self.structure)

end

-- Serialization
function Tester:save(path)
    local config = {
        batch_size    = self.batch_size,
        emb_vecs      = self.emb_vecs:float(),
        learning_rate = self.learning_rate,
        mem_dim       = self.mem_dim,
        reg           = self.reg,
        structure     = self.structure,
        task          = self.task,
        feats_dim     = self.feats_dim,
    }

    torch.save(path, {
        params = self.params,
        config = config,
    })
end