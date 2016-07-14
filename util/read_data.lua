--[[

  Functions for loading data from disk.

--]]

function utils.load_embedding(dir, vocab, uniform_val)
    local emb_prefix = dir .. '/glove/glove.840B'
    local emb_vocab = utils.Vocab(emb_prefix .. '.vocab')
    local emb_vecs = torch.load(emb_prefix .. '.300d.th')
    local emb_dim = emb_vecs:size(2)
    local num_unk = 0
    uniform_val = uniform_val or 0.05
    local vecs = torch.Tensor(vocab.size, emb_dim)
    for i = 1, vocab.size do
        local w = vocab:token(i)
        if emb_vocab:contains(w) then
            vecs[i] = emb_vecs[emb_vocab:index(w)]
        else
            num_unk = num_unk + 1
            vecs[i] = vecs[i]:uniform(-uniform_val, uniform_val)
        end
    end
    print('unknown words count = ' .. num_unk)
    emb_vecs = nil
    emb_vocab = nil
    collectgarbage()
    return vecs
end

function utils.read_embedding(vocab_path, emb_path)
    local vocab = utils.Vocab(vocab_path)
    local embedding = torch.load(emb_path)
    return vocab, embedding
end

function utils.read_sentences(path, vocab)
    local sentences = {}
    local file = io.open(path, 'r')
    local line
    while true do
        line = file:read()
        if line == nil then break end
        local tokens = stringx.split(line)
        local len = #tokens
        local sent = torch.IntTensor(len)
        for i = 1, len do
            local token = tokens[i]
            sent[i] = vocab:index(token)
        end
        sentences[#sentences + 1] = sent
    end

    file:close()
    return sentences
end

function utils.read_trees(parent_path, label_path)
    local parent_file = io.open(parent_path, 'r')
    local label_file
    if label_path ~= nil then label_file = io.open(label_path, 'r') end
    local count = 0
    local trees = {}

    while true do
        local parents = parent_file:read()
        if parents == nil then break end
        parents = stringx.split(parents)
        for i, p in ipairs(parents) do
            parents[i] = tonumber(p)
        end

        local labels
        if label_file ~= nil then
            labels = stringx.split(label_file:read())
            for i, l in ipairs(labels) do
                -- ignore unlabeled nodes
                if l == '#' then
                    labels[i] = nil
                else
                    labels[i] = tonumber(l)
                end
            end
        end
        count = count + 1
        trees[count] = utils.read_tree(parents, labels)
    end
    parent_file:close()
    return trees
end

function utils.read_tree(parents, labels)
    local size = #parents
    local trees = {}
    if labels == nil then labels = {} end
    local root
    for i = 1, size do
        if not trees[i] and parents[i] ~= -1 then
            local idx = i
            local prev
            while true do
                local parent = parents[idx]
                if parent == -1 then
                    break
                end

                local tree = utils.Tree()
                if prev ~= nil then
                    tree:add_child(prev)
                end
                trees[idx] = tree
                tree.idx = idx
                tree.gold_label = labels[idx]
                if trees[parent] ~= nil then
                    trees[parent]:add_child(tree)
                    break
                elseif parent == 0 then
                    root = tree
                    break
                else
                    prev = tree
                    idx = parent
                end
            end
        end
    end

    -- index leaves (only meaningful for constituency trees)
    local leaf_idx = 1
    for i = 1, size do
        local tree = trees[i]
        if tree ~= nil and tree.num_children == 0 then
            tree.leaf_idx = leaf_idx
            leaf_idx = leaf_idx + 1
        end
    end
    return root
end

function utils.read_dataset(dir, vocab)
    if string.find(dir, 'sick') then
        return utils.read_sick_dataset(dir, vocab)
    elseif string.find(dir, 'msrp') then
        return utils.read_msrp_dataset(dir, vocab)
    elseif string.find(dir, 'snli') then
        return utils.read_snli_dataset(dir, vocab)
    elseif string.find(dir, 'wqa') then
        return utils.read_wqa_dataset(dir, vocab)
    elseif string.find(dir, 'grade') then
        return utils.read_grade_dataset(dir, vocab)
    else
        error('No such dataset!')
    end
end

--[[

  Semantic Relatedness on SICK dataset

--]]

function utils.read_sick_dataset(dir, vocab)
    local dataset = {}
    dataset.vocab = vocab
    dataset.ltrees = utils.read_trees(dir .. 'a.parents')
    dataset.rtrees = utils.read_trees(dir .. 'b.parents')
    dataset.lsents = utils.read_sentences(dir .. 'a.toks', vocab)
    dataset.rsents = utils.read_sentences(dir .. 'b.toks', vocab)
    dataset.size = #dataset.ltrees
    local id_file = torch.DiskFile(dir .. 'id.txt')
    local sim_file = torch.DiskFile(dir .. 'sim.txt')
    dataset.ids = torch.IntTensor(dataset.size)
    dataset.labels = torch.Tensor(dataset.size)
    for i = 1, dataset.size do
        dataset.ids[i] = id_file:readInt()
        dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1)
    end
    id_file:close()
    sim_file:close()
    return dataset
end

--[[

  SNLI Dataset

--]]
function utils.read_snli_dataset(dir, vocab)
    local dataset = {}
    dataset.vocab = vocab
    dataset.ltrees = utils.read_trees(dir .. 'a.parents')
    dataset.rtrees = utils.read_trees(dir .. 'b.parents')

    dataset.lsents = utils.read_sentences(dir .. 'a.toks', vocab)
    dataset.rsents = utils.read_sentences(dir .. 'b.toks', vocab)
    dataset.size = #dataset.ltrees
    local label_file = torch.DiskFile(dir .. 'label.txt')
    dataset.labels = torch.Tensor(dataset.size)
    for i = 1, dataset.size do
        dataset.labels[i] = label_file:readInt()
    end
    label_file:close()
    return dataset
end

--[[

  Microsoft Research phrase detection Dataset

--]]

function utils.read_msrp_dataset(dir, vocab)
    local dataset = {}
    dataset.vocab = vocab
    local ltrees = utils.read_trees(dir .. 'a.parents')
    local rtrees = utils.read_trees(dir .. 'b.parents')
    local lsents = utils.read_sentences(dir .. 'a.toks', vocab)
    local rsents = utils.read_sentences(dir .. 'b.toks', vocab)
    local size = #ltrees

    local label_file = torch.DiskFile(dir .. 'label.txt')
    local labels = torch.Tensor(size)
    for i = 1, size do
        labels[i] = label_file:readInt() + 1
    end
    label_file:close()
    dataset.lsents = lsents
    dataset.ltrees = ltrees
    dataset.rsents = rsents
    dataset.rtrees = rtrees
    dataset.size = size
    dataset.labels = labels
    return dataset
end

function utils.split_data(dataset, split)

end

--[[

  WikiQA dataset

--]]

function utils.read_wqa_dataset(dir, vocab)
    local dataset = {}
    dataset.vocab = vocab
    dataset.ltrees = utils.read_trees(dir .. 'a.parents')
    dataset.rtrees = utils.read_trees(dir .. 'b.parents')
    dataset.lsents = utils.read_sentences(dir .. 'a.toks', vocab)
    dataset.rsents = utils.read_sentences(dir .. 'b.toks', vocab)
    dataset.size = #dataset.ltrees
    local label_file = torch.DiskFile(dir .. 'label.txt')
    local qid_file = torch.DiskFile(dir .. 'qid.txt')
    dataset.labels = torch.Tensor(dataset.size)
    dataset.qids = torch.Tensor(dataset.size)
    for i = 1, dataset.size do
        dataset.labels[i] = label_file:readInt() + 1
        dataset.qids[i] = qid_file:readInt() + 1
    end
    label_file:close()
    qid_file:close()
    return dataset
end

--[[

  AI2-8grade dataset

--]]
function utils.read_grade_dataset(dir, vocab)
    local dataset = {}
    dataset.vocab = vocab
    dataset.ltrees = utils.read_trees(dir .. 'a.parents')
    dataset.rtrees = utils.read_trees(dir .. 'b.parents')
    dataset.lsents = utils.read_sentences(dir .. 'a.toks', vocab)
    dataset.rsents = utils.read_sentences(dir .. 'b.toks', vocab)
    dataset.size = #dataset.ltrees
    local label_file = torch.DiskFile(dir .. 'label.txt')
    dataset.labels = torch.Tensor(dataset.size)
    for i = 1, dataset.size do
        dataset.labels[i] = label_file:readInt() + 1
    end
    label_file:close()
    return dataset
end

function utils.split_data(dataset, ratio)
    local dset_test = {}
    local size = math.floor(dataset.size * ratio)
    local indeics = torch.randperm(dataset.size)
    dset_test.lsents = {}
    dset_test.rsents = {}
    dset_test.ltrees = {}
    dset_test.rtrees = {}
    dset_test.labels = torch.zeros(size)
    dset_test.size = size
    for i = 1, size do
        dset_test.lsents[i] = dataset.lsents[indeics[i]]
        dset_test.rsents[i] = dataset.rsents[indeics[i]]
        dset_test.ltrees[i] = dataset.ltrees[indeics[i]]
        dset_test.rtrees[i] = dataset.rtrees[indeics[i]]
        dset_test.labels[i] = dataset.labels[indeics[i]]
    end
    local dset_train = {}
    dset_train.lsents = {}
    dset_train.rsents = {}
    dset_train.ltrees = {}
    dset_train.rtrees = {}
    dset_train.labels = torch.zeros(dataset.size - size)
    dset_train.size = dataset.size - size
    for i = size + 1, dataset.size do
        dset_train.lsents[i - size] = dataset.lsents[indeics[i]]
        dset_train.rsents[i - size] = dataset.rsents[indeics[i]]
        dset_train.ltrees[i - size] = dataset.ltrees[indeics[i]]
        dset_train.rtrees[i - size] = dataset.rtrees[indeics[i]]
        dset_train.labels[i - size] = dataset.labels[indeics[i]]
    end
    return dset_train, dset_test
end