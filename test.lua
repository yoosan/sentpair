--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 6/29/16, 2016.
 - Licence MIT
--]]
require 'nn'
require 'nngraph'

local lvec, rvec, inputs, input_dim, extra_feat

input_dim = 2 * 10
local linput, rinput = nn.Identity()(), nn.Identity()()
lvec, rvec = linput, rinput
inputs = { linput, rinput }


local mult_dist = nn.CMulTable() { lvec, rvec }
local add_dist = nn.Abs()(nn.CSubTable() { lvec, rvec })
local vec_dist_feats = nn.JoinTable(1) { mult_dist, add_dist }
local vecs_to_input = nn.gModule(inputs, { vec_dist_feats })

local a, b = torch.ones(10), torch.ones(10)
local res = vecs_to_input:forward({a, b})

local temp = nn.Sequential()
:add(vecs_to_input)
:add(nn.Linear(input_dim, 5))
:add(nn.Sigmoid())

local res2 = temp:forward({a, b})
print(res2)

local i = nn.Identity()()
local j = nn.JoinTable(1)({nn.Identity()(temp), i})
local ss = nn.gModule({inputs, i}, j)

local res3 = ss:forward({{a, b}, torch.zeros(3)})
print(res3)
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