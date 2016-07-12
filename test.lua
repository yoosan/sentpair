--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 6/29/16, 2016.
 - Licence MIT
--]]
require 'init.lua'

--local lvec, rvec, inputs, input_dim, extra_feat
--
--input_dim = 2 * 10
--local linput, rinput = nn.Identity()(), nn.Identity()()
--lvec, rvec = linput, rinput
--inputs = { linput, rinput }
--
--
--local mult_dist = nn.CMulTable() { lvec, rvec }
--local add_dist = nn.Abs()(nn.CSubTable() { lvec, rvec })
--local vec_dist_feats = nn.JoinTable(1) { mult_dist, add_dist }
--local vecs_to_input = nn.gModule(inputs, { vec_dist_feats })
--
--local a, b = torch.ones(10), torch.ones(10)
--local res = vecs_to_input:forward({a, b})
--
--local temp = nn.Sequential()
--:add(vecs_to_input)
--:add(nn.Linear(input_dim, 5))
--:add(nn.Sigmoid())
--
--local res2 = temp:forward({a, b})
--print(res2)
--
--local i = nn.Identity()()
--local j = nn.JoinTable(1)({nn.Identity()(temp), i})
--local ss = nn.gModule({inputs, i}, j)
--
--local res3 = ss:forward({{a, b}, torch.zeros(3)})
--print(res3)
--local classifier, feats_dim, lr_out
--if self.task == 'MSRP' or self.task == 'WQA' then
--    classifier = nn.Sigmoid()
--    feats_dim = self.feats_dim + self.extra_dim
--    lr_out = parallel
--elseif self.task == 'SICK' or self.task == 'SNLI' then
--    feats_dim = self.feats_dim
--    classifier = nn.LogSoftMax()
--    lr_out = temp
--end
--
--local output_module = nn.Sequential()
--:add(lr_out)
--:add(nn.Linear(feats_dim, self.num_classes))
--:add(classifier)

local data_dir = 'data/msrp'
local vocab = utils.Vocab(data_dir .. '/vocab-cased.txt')
local dset_test = utils.read_dataset(data_dir .. '/test/', vocab)

function generate_plot_data(dataset)
    local len_stats = {}
    for i = 1, dataset.size do
        local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
        local unigram = nlptools.unigram(lsent, rsent)
        local bigram = nlptools.bigram(lsent, rsent)
        local trigram = nlptools.trigram(lsent, rsent)
        local tot = math.floor(50 * (unigram + bigram + trigram) / (lsent:size(1) + rsent:size(1)))
        if len_stats[tot] then
            len_stats[tot][1] = len_stats[tot][1] + 1
            table.insert(len_stats[tot][2], i)
        else
            len_stats[tot] = { 1, { i } }
        end
    end
    return len_stats
end

local prefix = 'GRADE-lstm'

local len_stats = generate_plot_data(dset_test)
local plot_data = {}
local load = torch.load('data/saved/MSRP-treegru.t7')
local preds, labels = load[1], load[2]
for k, v in pairs(len_stats) do
    local ids = v[2]
    if v[1] > 5 then
        local sub_preds = torch.zeros(#ids)
        local sub_gts = torch.zeros(#ids)
        for i = 1, #ids do
            sub_preds[i] = preds[ids[i]]
            sub_gts[i] = labels[ids[i]]
        end
        local acc = stats.accuracy(sub_preds, sub_gts)
        if plot_data[k] then
            plot_data[k] = plot_data[k] + acc
        else
            plot_data[k] = acc
        end
    end
end
print(#plot_data)

local pre_data = {}

for i = 2, #plot_data - 12, 2 do
    local prev = plot_data[10 + i - 1]
    local cur = plot_data[10 + i]
    local next = plot_data[10 + i + 1]
    pre_data[10 + i] = (prev + cur + next) / 3
end

print(pre_data)

torch.save('data/plot/MSRP-treegru-plot-ngram.t7', pre_data)


--local weights = torch.load('data/saved/weights/wei_1.t7')
--print(weights)

--local data = torch.load('data/saved/MSRP-atreelstm.t7')
--
--print(stats.accuracy(data[1], data[2]))
--
--local wei = torch.load('data/saved/weights/wei_msrp1660.t7')
--print(wei)

--for i = 1, data[1]:size(1) do
--    if data[1][i] == data[2][i] then
--        print(i)
--    end
--end