import torchfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

minorLocator = MultipleLocator(1)

task = ['SICK', 'GRADE']
lstm_data = torchfile.load('data/plot/MSRP-lstm-plot-ngram.t7')
gru_data = torchfile.load('data/plot/MSRP-gru-plot-ngram.t7')
treelstm_data = torchfile.load('data/plot/MSRP-treelstm-plot-ngram.t7')
treegru_data = torchfile.load('data/plot/MSRP-treegru-plot-ngram.t7')
atreelstm_data = torchfile.load('data/plot/MSRP-atreelstm-plot-ngram.t7')
atreegru_data = torchfile.load('data/plot/MSRP-atreegru-plot-ngram.t7')

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

x_axis = x_lstm[3:-1]
y_lstm = y_lstm[3:-1]
y_gru = y_gru[3:-1]
y_treelstm = y_treelstm[3:-1]
y_treegru = y_treegru[3:-1]
y_atreelstm = y_atreelstm[3:-1]
y_atreegru = y_atreegru[3:-1]

# fig, ax = plt.subplots(figsize=(15, 10))
fig, ax = plt.subplots()
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_xlabel('c * (ngrams / mean_sent_length)')
ax.set_ylabel('accuracy')
ax.xaxis.set_minor_locator(minorLocator)


l2 = ax.plot(x_axis, y_gru, marker='o', color='green', label='Seq-GRUs')
l1 = ax.plot(x_axis, y_lstm, marker='o', color='blue', label='Seq-LSTMs')

l4 = ax.plot(x_axis, y_treegru, marker='v', color='orange', label='Tree-GRUs')
l3 = ax.plot(x_axis, y_treelstm, marker='v', color='gray', label='Tree-LSTMs')

l6 = ax.plot(x_axis, y_atreegru, marker='d', color='red', label='Attentive Tree-GRUs')
l5 = ax.plot(x_axis, y_atreelstm, marker='d', color='purple', label='Attentive Tree-LSTMs')

ax.legend(loc="lower right", shadow=False, fancybox=True)
ax.grid()
# ax.yaxis.grid(True)
plt.show()
# plt.savefig('grade.eps', format='eps', dpi=1000)
