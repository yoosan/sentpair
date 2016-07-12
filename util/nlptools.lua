--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 6/27/16, 2016.
 - Licence MIT
--]]

nlptools = {}

function table.equal(a, b)
    assert(#a == #b, 'table size should same')
    for i = 1, #a do
        if a[i] ~= b[i] then
            return false
        end
    end
    return true
end

function table.contains(a, ele)
    for i = 1, #a do
        if type(ele) == 'table' then
            if table.equal(ele, a[i]) then
                return true
            end
        elseif a[i] == ele then
            return true
        end
    end
    return false
end

function utils.makeHashTable(tb)
    local res = {}
    for _, v in pairs(tb) do
        if #res > 0 and table.contains(res, v) then
        else
            table.insert(res,v)
        end
    end
    return res
end

-- a,b: tensor of the sentence indices, such as [233, 125, 32, 1] and [125, 32, 332]
function nlptools.unigram(a, b)
    local tb_a = torch.totable(a)
    local tb_b = torch.totable(b)
    local dict_a = utils.makeHashTable(tb_a)
    local dict_b = utils.makeHashTable(tb_b)
    local cnt = 0
    for i = 1, #dict_a do
        for j = 1, #dict_b do
            if dict_a[i] == dict_b[j] then
                cnt = cnt + 1
            end
        end
    end
    return cnt
end

function nlptools.makeBigram(a)
    local bigram_tb = {}
    for i = 1, a:size(1) - 1 do
        table.insert(bigram_tb, {a[i], a[i+1]})
    end
    return bigram_tb
end

function nlptools.makeTrigram(a)
    local trigram_tb = {}
    for i = 1, a:size(1) - 2 do
        table.insert(trigram_tb, {a[i], a[i+1], a[i+2]})
    end
    return trigram_tb
end

function nlptools.bigram(a, b)
    local bigram_a = nlptools.makeBigram(a)
    local bigram_b = nlptools.makeBigram(b)
    local prd_bigram_a = utils.makeHashTable(bigram_a)
    local prd_bigram_b = utils.makeHashTable(bigram_b)
    local cnt = 0
    for i = 1, #prd_bigram_a do
        for j = 1, #prd_bigram_b do
            if table.equal(prd_bigram_a[i], prd_bigram_b[j]) then
                cnt = cnt + 1
            end
        end
    end
    return cnt
end

function nlptools.trigram(a, b)
    local trigram_a = nlptools.makeTrigram(a)
    local trigram_b = nlptools.makeTrigram(b)
    local prd_trigram_a = utils.makeHashTable(trigram_a)
    local prd_trigram_b = utils.makeHashTable(trigram_b)
    local cnt = 0
    for i = 1, #prd_trigram_a do
        for j = 1, #prd_trigram_b do
            if table.equal(prd_trigram_a[i], prd_trigram_b[j]) then
                cnt = cnt + 1
            end
        end
    end
    return cnt
end

--local a = {2, 3, 3, 1, 5, 1, 5 }
--local at = torch.Tensor(a)
--local b = {3, 2, 4, 1, 5}
--local bt = torch.Tensor(b)
--local cnt = nlptools.unigram(torch.Tensor(a), torch.Tensor(b))
--
--print(nlptools.bigram(at, bt))
--
--print(nlptools.trigram(at, bt))