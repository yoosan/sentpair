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

printf('size of vocab = %d\n', vocab.size)
printf('number of train = %d\n', dset_train.size)
printf('number of dev   = %d\n', dset_dev.size)
printf('number of test  = %d\n', dset_test.size)

-- train and evaluate
local trainer = Trainer(config)
trainer:print_config()
--trainer:train(dset_train)
--local predictions = trainer:eval(dset_dev)
--local pearson_score = stats.pearson(predictions, dset_dev.labels)
--local spearman_score = stats.spearmanr(predictions, dset_dev.labels)
--local mse_score = stats.mse(predictions, dset_dev.labels)
--printf('-- Dev pearson = %.4f, spearmanr = %.4f, mse = %.4f \n',
--    pearson_score, spearman_score, mse_score)

function run(tr, n_epoches, dset_train, dset_dev, dset_test)
    header('Training model ... ')
    local train_start = sys.clock()
    local best_score = -1.0
    local best_params
    local best_trainer = tr
    for i = 1, n_epoches do
        local start = sys.clock()
        printf('-- epoch %d \n', i)
        tr:train(dset_train)
        printf('-- finished epoch in %.2fs\n', sys.clock() - start)
        local predictions = tr:eval(dset_dev)
        local dev_score
        if tr.task == 'SICK' then
            local pearson_score = stats.pearson(predictions, dset_dev.labels)
            local spearman_score = stats.spearmanr(predictions, dset_dev.labels)
            local mse_score = stats.mse(predictions, dset_dev.labels)
            printf('-- Dev pearson = %.4f, spearmanr = %.4f, mse = %.4f \n',
                pearson_score, spearman_score, mse_score)
            dev_score = pearson_score
        elseif tr.task == 'MSRP' then
            local accuracy = stats.accuracy(predictions, dset_dev.labels)
            local f1 = stats.f1(predictions, dset_dev.labels)
            printf('-- Dev accuracy = %.4f, f1 score = %.4f \n', accuracy, f1)
            dev_score = accuracy
        else
            local accuracy = stats.accuracy(predictions, dset_dev.labels)
            printf('-- Dev accuracy = %.4f \n', accuracy)
            dev_score = accuracy
        end
        if dev_score > best_score then
            best_score = dev_score
            best_trainer.params:copy(tr.params)
        end
    end
    printf('finished training in %.2fs\n', sys.clock() - train_start)
    header('Evaluating on test set')
    printf('-- using model with dev score = %.4f\n', best_score)
    local test_preds = best_trainer:eval(dset_test)
    if tr.task == 'SICK' then
        local pearson_score = stats.pearson(test_preds, dset_test.labels)
        local spearman_score = stats.spearmanr(test_preds, dset_test.labels)
        local mse_score = stats.mse(test_preds, dset_test.labels)
        printf('-- Test pearson = %.4f, spearmanr = %.4f, mse = %.4f \n',
            pearson_score, spearman_score, mse_score)
    elseif tr.task == 'MSRP' then
        local accuracy = stats.accuracy(test_preds, dset_test.labels)
        local f1 = stats.f1(test_preds, dset_test.labels)
        printf('-- Test accuracy = %.4f, f1 score = %.4f \n', accuracy, f1)
    else
        local accuracy = stats.accuracy(test_preds, dset_test.labels)
        printf('-- Test accuracy = %.4f \n', accuracy)
    end
end

run(trainer, config.n_epoches, dset_train, dset_dev, dset_test)

