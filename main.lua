--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 6/21/16, 2016.
 - Licence MIT
--]]

require 'init.lua'

local cmd = torch.CmdLine()
cmd:option('-task', 'SICK', 'training task, dataset for modeling sentence pair')
cmd:option('-structure', 'lstm', 'model structure')
cmd:option('-mem_dim', 150, 'dimension of memory')
cmd:option('-n_epoches', 10, 'number of epoches for training')
cmd:option('-lr', 0.05, 'learning rate')
cmd:option('-batch_size', 25, 'batch size')
cmd:option('-feats_dim', 50, 'features dimensions')
local config = cmd:parse(arg or {})

header(config.task .. ' dataset for modeling sentence pair')

-- load word embedding and dataset
local data_dir = 'data/' .. config.task:lower()
local vocab = utils.Vocab(data_dir .. '/vocab-cased.txt')
local emb_vecs = utils.load_embedding('data', vocab)
config.emb_vecs = emb_vecs

local dset_train = utils.read_dataset(data_dir .. '/train/', vocab)
local dset_dev = utils.read_dataset(data_dir .. '/dev/', vocab)
local dset_test = utils.read_dataset(data_dir .. '/test/', vocab)

printf('number of train = %d\n', dset_train.size)
printf('number of dev   = %d\n', dset_dev.size)
printf('number of test  = %d\n', dset_test.size)

local trainer = Trainer(config)
trainer:train(dset_train)

