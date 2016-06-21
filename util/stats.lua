--[[
 - Author: yoosan, SYSUDNLP Group
 - Date: 15/5/8, 2015.
 - Licence MIT
--]]

--[[
-- Statistics and Math Library
-- math.pearson
-- math.spearman
-- stats.pearson
-- stats.spearman
-- stats.accuracy
-- stats.mse
--]]

-- Description: Pearson correlation coefficient
-- Input: a is an n*2 table
function math.pearson(a)
    -- compute the mean
    local x1, y1 = 0, 0
    for _, v in pairs(a) do
        x1, y1 = x1 + v[1], y1 + v[2]
    end
    -- compute the coefficient
    x1, y1 = x1 / #a, y1 / #a
    local x2, y2, xy = 0, 0, 0
    for _, v in pairs(a) do
        local tx, ty = v[1] - x1, v[2] - y1
        xy, x2, y2 = xy + tx * ty, x2 + tx * tx, y2 + ty * ty
    end
    return xy / math.sqrt(x2) / math.sqrt(y2)
end

-- Description: Spearman correlation coefficient
function math.spearman(a)
    local function aux_func(t) -- auxiliary function
    return (t == 1 and 0) or (t * t - 1) * t / 12
    end

    for _, v in pairs(a) do v.r = {} end
    local T, S = {}, {}
    -- compute the rank
    for k = 1, 2 do
        table.sort(a, function(u, v) return u[k] < v[k] end)
        local same = 1
        T[k] = 0
        for i = 2, #a + 1 do
            if i <= #a and a[i - 1][k] == a[i][k] then same = same + 1
            else
                local rank = (i - 1) * 2 - same + 1
                for j = i - same, i - 1 do a[j].r[k] = rank end
                if same > 1 then T[k], same = T[k] + aux_func(same), 1 end
            end
        end
        S[k] = aux_func(#a) - T[k]
    end
    -- compute the coefficient
    local sum = 0
    for _, v in pairs(a) do -- TODO: use nested loops to reduce loss of precision
    local t = (v.r[1] - v.r[2]) / 2
    sum = sum + t * t
    end
    return (S[1] + S[2] - sum) / 2 / math.sqrt(S[1] * S[2])
end

-- Pearson correlation
function stats.pearson(x, y)
    x = x - x:mean()
    y = y - y:mean()
    return x:dot(y) / (x:norm() * y:norm())
end

-- sparman correlation
function stats.spearmanr(x, y)
    assert(x:size(1) == y:size(1))
    local size = x:size(1)
    local input = {}
    for i = 1, size do
        table.insert(input, { x[i], y[i] })
    end
    return math.spearman(input)
end

-- mean squared error for semantic relatedness
function stats.mse(x, y)
    local _y = y * 4 + 1
    return ((x - _y):norm() ^ 2) / x:size(1)
end

-- accuracy rate
function stats.accuracy(x, y)
    assert(x:size(1) == y:size(1), '#predictions == #ground truth')
    return torch.eq(x, y):sum() / x:size(1)
end

-- precision
function stats.precision(x, y)
    local tp = 0
    local fp = 0
    local fn = 0
    local tn = 0
    for i = 1, x:size(1) do
        if y[i] == 2 and x[i] == 2 then
            tp = tp + 1
        elseif y[i] == 2 and x[i] == 1 then
            fn = fn + 1
        elseif y[i] == 1 and x[i] == 2 then
            fp = fp + 1
        elseif y[i] == 1 and x[i] == 1 then
            tn = tn + 1
        end
    end
    return tp / (tp + fp)
end

function stats.recall(x, y)
    local tp = 0
    local fp = 0
    local fn = 0
    local tn = 0
    for i = 1, x:size(1) do
        if y[i] == 2 and x[i] == 2 then
            tp = tp + 1
        elseif y[i] == 2 and x[i] == 1 then
            fn = fn + 1
        elseif y[i] == 1 and x[i] == 2 then
            fp = fp + 1
        elseif y[i] == 1 and x[i] == 1 then
            tn = tn + 1
        end
    end
    return tp / (tp + fn)
end

function stats.f1(x, y)
    local precision = stats.precision(x, y)
    local recall = stats.recall(x, y)
    return (2 * precision * recall) / (precision + recall)
end

-- argmax function
function stats.argmax(v)
    local idx = 1
    local max = v[1]
    for i = 2, v:size(1) do
        if v[i] > max then
            max = v[i]
            idx = i
        end
    end
    return idx
end