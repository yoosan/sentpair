--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 10/21/15, 2015.
 - Licence MIT
--]]

local Trainer = torch.class('Trainer')

function Trainer:__init(config)
    self.task          = config.task -- SICK, SNLI, MSRP, WQA
    self.mem_dim       = config.mem_dim       or 150
    self.learning_rate = config.learning_rate or 0.05
    self.batch_size    = config.batch_size    or 25
    self.num_layers    = config.num_layers    or 1
    self.reg           = config.reg           or 1e-4
    self.structure     = config.structure     or 'lstm' -- gru, treelstm, treegru, atreelstm, atreegru
    self.feats_dim     = config.feats_dim     or 50

    -- word embedding
    self.emb_vecs = config.emb_vecs
    self.emb_dim = config.emb_vecs:size(2)

    -- optimizer config
    self.optim_func = config.optim_func or optim.adagrad
    self.optim_state = {
        learning_rate = self.learning_rate,

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
        error('Invalid structure.')
    end

    share_params(self.rmodel, self.lmodel)

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

end

function Trainer:new_output_module()
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

function Trainer:train(dataset)
    self.modules:training()

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
                local lrep, rrep
                if self.structure == 'lstm' or self.structure == 'gru' then
                    lrep = self.lmodel:forward(linputs)
                    rrep = self.rmodel:forward(rinputs)
                elseif self.sturcture == 'treelstm' or self.sturcture == 'treegru' then
                    lrep = self.model:forward(ltree, linputs)
                    rrep = self.model:forward(rtree, rinputs)
                else
                    local seq_lrep = self.latt:forward(linputs)
                    local seq_rrep = self.ratt:forward(rinputs)
                    lrep = self.model:forward(ltree, linputs, seq_lrep)
                    rrep = self.model:forward(rtree, rinputs, seq_rrep)
                end

                -- feed to the output module and compute loss
                local output = self.output_module:forward({lrep, rrep})
                local example_loss = self.criterion:forward(output, targets[j])
                loss = loss + example_loss

                -- backward
                
            end
        end

    end

end