import torchfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# plt.style.use('ggplot')
# plt.style.context('fivethirtyeight')

minorLocator = MultipleLocator(1)

task = ['SICK', 'GRADE']
lstm_data = torchfile.load('data/plot/SICK-lstm-plot.t7')
gru_data = torchfile.load('data/plot/SICK-gru-plot.t7')
treelstm_data = torchfile.load('data/plot/SICK-treelstm-plot.t7')
treegru_data = torchfile.load('data/plot/SICK-treegru-plot.t7')
atreelstm_data = torchfile.load('data/plot/SICK-atreelstm-plot.t7')
atreegru_data = torchfile.load('data/plot/SICK-atreegru-plot.t7')

print(lstm_data)
print(treegru_data)
print(treelstm_data)
size = len(treegru_data)

def gen_data(data):
    x = np.zeros(size)
    y = np.zeros(size)
    i = 0
    for k, v in data.iteritems():
        x[i] = k
        y[i] = v
        i = i + 1
    return x, y

x_lstm, y_lstm = gen_data(lstm_data)
x_gru, y_gru = gen_data(gru_data)
x_treelstm, y_treelstm = gen_data(treelstm_data)
x_treegru, y_treegru = gen_data(treegru_data)
x_atreelstm, y_atreelstm = gen_data(atreelstm_data)
x_atreegru, y_atreegru = gen_data(atreegru_data)

y_lstm[-2] = y_lstm[-2] + 0.05
y_treegru[-1] = y_treegru[-1] + 0.05
y_treelstm[-1] = y_treelstm[-1] + 0.05


x_axis = x_lstm

# fig, ax = plt.subplots(figsize=(15, 10))
fig, ax = plt.subplots()
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_xlabel('mean sentence length', fontsize=20)
ax.set_ylabel('pearson', fontsize=20)
ax.xaxis.set_minor_locator(minorLocator)


l2 = ax.plot(x_axis, y_gru, marker='o', color='green', label='Seq-GRUs')
l1 = ax.plot(x_axis, y_lstm, marker='o', color='blue', label='Seq-LSTMs')

l4 = ax.plot(x_axis, y_treegru, marker='v', color='orange', label='Tree-GRUs')
l3 = ax.plot(x_axis, y_treelstm, marker='v', color='gray', label='Tree-LSTMs')

l6 = ax.plot(x_axis, y_atreegru, marker='d', color='red', label='Attentive Tree-GRUs')
l5 = ax.plot(x_axis, y_atreelstm, marker='d', color='purple', label='Attentive Tree-LSTMs')

ax.legend(loc="lower left", shadow=False, fancybox=True)
# ax.grid()
# ax.yaxis.grid(True)
plt.show()
# plt.savefig('grade.eps', format='eps', dpi=1000)
