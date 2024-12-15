import math
import mne
import numpy as np
from matplotlib import pyplot as plt

distance_dict = {'64': 0.025, '32': 0.034, '16': 0.034}
channel_num = 64
pos = np.load('pos.npy')

add_dict = {'AFz', 'POz', 'Fz', }
biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
channel64_names = biosemi64_montage.ch_names
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
channel_names = biosemi_montage.ch_names
biosemi_montage.plot(show_names=True)
channel_index = [channel64_names.index(channel) for channel in channel_names]

pos = pos[channel_index, :]
# 绘制电极坐标
plt.figure(dpi=300)

plt.axis('equal')

plt.axis('off')
X_data, Y_data = [], []
for k_ch in range(channel_num):
    X_data.append(pos[k_ch][0])
    Y_data.append(pos[k_ch][1])
    plt.text(pos[k_ch][0], pos[k_ch][1], channel64_names[k_ch], fontsize=4, verticalalignment='center', color='black',
             horizontalalignment='center', zorder=3)
    # # 绘制电极圆圈
    circle = plt.Circle(pos[k_ch], 0.004, color='limegreen', fill=True, linewidth=0.4, zorder=2)
    plt.gcf().gca().add_artist(circle)


# 求连接权重前32大的边
adj = np.load('adjs0.npy')
d = adj.flatten()
x = abs(np.sort(-d))
index = np.where(adj>x[128])
index = np.array(index)
print(np.shape(index))
# plt.show()
num = len(index[0])
edges = []
for i in range(num):
    x = [pos[index[0, i]][0], pos[index[1, i]][0]]
    y = [pos[index[0, i]][1], pos[index[1, i]][1]]
    plt.plot(x, y, zorder=1, color='black', linewidth=0.04)

# np.save(f'edges{channel_num}.npy', edges)
plt.show()