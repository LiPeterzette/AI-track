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
# biosemi_montage.plot(show_names=True)
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
adj = np.load('../channel_score/adj_S3_64_KUL.npy')


# channel_select = attention_score.argsort()[32:]
plt.gcf().subplots_adjust(left=0.05,top=0.91,bottom=0.09)

channel_select = []
score = np.zeros([64])
for j in range(64):
    adj[j, j] = 0
for i in range(1, 100):
    i = np.argmax(adj)
    a, b = divmod(i, adj.shape[1])
    adj[a, b] = 0
    adj[b, a] = 0
    if a in channel_select:
        score[a] = score[a] + 1
    else:
        channel_select.append(a)
    if b in channel_select:
        score[b] = score[b] + 1
    else:
        channel_select.append(b)
    x = [pos[a][0], pos[b][0]]
    y = [pos[a][1], pos[b][1]]
    plt.plot(x, y, zorder=1, color='black')
    l = len(channel_select)
    if l >= 16:
        break
edges = []
for cha1 in range(channel_num):
    for cha2 in range(channel_num):
        is_connect = False



        # 将连接关系添加到edges中
        if is_connect:
            edges.append([cha1, cha2])
            x = [pos[cha1][0], pos[cha2][0]]
            y = [pos[cha1][1], pos[cha2][1]]
            plt.plot(x, y, zorder=1, color='black',linewidth=10)

# np.save(f'edges{channel_num}.npy', edges)
plt.show()
