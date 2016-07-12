import matplotlib.pyplot as plt
import torchfile
import numpy as np

with open('data/msrp/test/a.toks') as fa, open('data/msrp/test/b.toks') as fb:
    sa = fa.readlines()
    sb = fb.readlines()

a = sa[1659]
b = sb[1659]
wei = torchfile.load('data/saved/weights/wei_msrp1660.t7')

print(a)
print(b)
w = wei[1]
print(wei[0])
print(w)
column_labels = b.split(' ')
row_labels = column_labels
data = np.zeros((len(row_labels), len(row_labels)))

data[1][2] = w[-1][0][0]
data[4][2] = w[-1][1][0]
data[14][2] = w[-1][2][0]
data[0][1] = 1
data[3][4] = w[-3][0][0]
data[5][4] = w[-3][1][0]
data[7][5] = 1
data[6][7] = w[-5][0][0]
data[9][7] = w[-5][1][0]
data[8][9] = w[-6][0][0]
data[10][9] = w[-6][1][0]
data[11][9] = w[-6][2][0]
data[13][11] = 1
data[12][13] = 1

fig, ax = plt.subplots()
# put the major ticks at the middle of each cell
plt.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
# ax.xaxis.tick_top()

ax.set_xticklabels(row_labels, minor=False, rotation='vertical')
ax.set_yticklabels(column_labels, minor=False)
plt.colorbar()
plt.show()
