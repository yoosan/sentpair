require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

advnn = {}
stats = {}
printf = utils.printf
utils = {}

torch.setnumthreads(16)

include('util/read_data.lua')
include('util/Tree.lua')
include('util/Vocab.lua')
include('util/stats.lua')
include('layers/CRowAddTable.lua')
include('models/LSTM.lua')
include('models/GRU.lua')
include('models/TreeGRU.lua')
include('models/TreeLSTM.lua')
include('models/ChildSumTreeLSTM.lua')
include('models/ChildSumTreeGRU.lua')
include('models/AttTreeGRU.lua')
include('models/AttTreeLSTM.lua')

-- global paths (modify if desired)
utils.data_dir        = 'data'
utils.models_dir      = 'trained_models'
utils.predictions_dir = 'predictions'

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end
