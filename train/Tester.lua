--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 10/21/15, 2015.
 - Licence MIT
--]]

local Tester = torch.class('Tester')

function Tester:__init(config)
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
    if self.task == 'MSRP' or self.task == 'WQA' or self.task == 'GRADE' then
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
        self.model = nn.ChildSumTreeGRU(model_config)
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

    -- output module and feats modules
    self.output_module = self:new_output_module()

    -- modules
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

function Tester:predict(lsent, rsent, ltree, rtree)
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
    elseif self.structure == 'atreelstm' then
        l_seqrep = self.latt:forward(linputs)
        r_seqrep = self.ratt:forward(rinputs)
        inputs = {
            self.model:forward(ltree, linputs, r_seqrep)[2],
            self.model:forward(rtree, rinputs, l_seqrep)[2]
        }
    else
        l_seqrep = self.latt:forward(linputs)
        r_seqrep = self.ratt:forward(rinputs)
        inputs = {
            self.model:forward(ltree, linputs, r_seqrep),
            self.model:forward(rtree, rinputs, l_seqrep)
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

function Tester:eval(dataset)
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

function Tester:load(path)
    local misc = torch.load(path)
    self.config = misc.config
    self.params:copy(misc.params)
end