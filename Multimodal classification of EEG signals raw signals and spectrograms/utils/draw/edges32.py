import math
import mne
import numpy as np
from matplotlib import pyplot as plt

distance_dict = {'64': 0.025, '32': 0.034, '16': 0.034}
channel_num = 64
pos = np.load('../pos.npy')
# channel_index = less_channel(channel_num)

biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
channel64_names = biosemi64_montage.ch_names


biosemi_montage = mne.channels.make_standard_montage(f'biosemi{channel_num}')
channel_names = biosemi_montage.ch_names
biosemi_montage.plot(show_names=True)

channel_index = [channel64_names.index(channel) for channel in channel_names]
pos = pos[channel_index, :]
# 绘制电极坐标
plt.figure()
plt.title(f'Graph(biosemi{channel_num})')

# fig = plt.figure(dpi=300, figsize=(10, 10))
# plt.axis('equal')
# xy_scale = 2
# plt.xlim(-xy_scale, xy_scale)
# plt.ylim(-xy_scale, xy_scale)
# plt.axis('off')
X_data, Y_data = [], []
for k_ch in range(channel_num):
    X_data.append(pos[k_ch][0])
    Y_data.append(pos[k_ch][1])
    # plt.text(pos[k_ch][0], pos[k_ch][1], k_ch, fontsize=4, family='Time New Roman', verticalalignment='center',
    #          horizontalalignment='center')
    # # 绘制电极圆圈
    # circle = plt.Circle(pos[0:2], 0.05, color='black', fill=False, linewidth=0.5)
    # plt.gcf().gca().add_artist(circle)
plt.scatter(X_data, Y_data)


#
# # 绘制外圆
# circle = plt.Circle([0, 0], 0.95, color='black', fill=False)
# plt.gcf().gca().add_artist(circle)



edges = []
for cha1 in range(channel_num):
    for cha2 in range(channel_num):
        is_connect = False
        # 距离
        if math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(pos[cha1], pos[cha2])])) < distance_dict[f'{channel_num}']:
            is_connect = True
        if channel_num == 32:
            # # 添加特殊连接
            if channel_names[cha1] == 'Fp1' and channel_names[cha2] in 'Fp2':
                is_connect = True
            if channel_names[cha1] == 'AF3' and channel_names[cha2] in ['AF4']:
                is_connect = True
            if channel_names[cha1] == 'Fz' and channel_names[cha2] in ['F3', 'F4']:
                is_connect = True
            if channel_names[cha1] == 'Pz' and channel_names[cha2] in ['P4', 'P3']:
                is_connect = True
            if channel_names[cha1] == 'Cz' and channel_names[cha2] in ['C4', 'C3']:
                is_connect = True

            # 删除边缘连接
            if channel_names[cha1] in ['P3', 'P4'] and channel_names[cha2] in ['O1', 'O2']:
                is_connect = False
            if channel_names[cha2] in ['F3', 'F4'] and channel_names[cha1] in ['Fp1', 'Fp2']:
                is_connect = False
            if channel_names[cha1] in ['P3', 'P4'] and channel_names[cha2] in ['O1', 'O2']:
                is_connect = False
            if channel_names[cha2] in ['F3', 'F4'] and channel_names[cha1] in ['Fp1', 'Fp2']:
                is_connect = False
        elif channel_num == 16:
            # 添加特殊连接
            if channel_names[cha1] == 'Fp1' and channel_names[cha2] in 'Fp2':
                is_connect = True
            # if channel_names[cha1] == 'AF3' and channel_names[cha2] in ['AF4']:
            #     is_connect = True
            if channel_names[cha1] == 'Fz' and channel_names[cha2] in ['F3', 'F4']:
                is_connect = True
            if channel_names[cha1] == 'Pz' and channel_names[cha2] in ['P3', 'P4']:
                is_connect = True
            if channel_names[cha1] == 'C3' and channel_names[cha2] in ['F3', 'P3']:
                is_connect = True
            if channel_names[cha1] == 'C4' and channel_names[cha2] in ['F4', 'P4']:
                is_connect = True
            if channel_names[cha1] == 'Cz' and channel_names[cha2] in ['C4', 'C3', 'Fz', 'Pz']:
                is_connect = True
            if channel_names[cha1] == 'T7' and channel_names[cha2] in ['F3', 'P3']:
                is_connect = True
            if channel_names[cha1] == 'T8' and channel_names[cha2] in ['F4', 'P4']:
                is_connect = True
        else:
            # 添加特殊连接
            if channel_names[cha1] == 'AFz' and channel_names[cha2] in ['AF3', 'AF4']:
                is_connect = True
            if channel_names[cha1] == 'POz' and channel_names[cha2] in ['PO3', 'PO4']:
                is_connect = True

            # 删除边缘连接
            if channel_names[cha1] in ['P5', 'P6'] and channel_names[cha2] in ['P9', 'P10']:
                is_connect = False
            if channel_names[cha2] in ['P5', 'P6'] and channel_names[cha1] in ['P9', 'P10']:
                is_connect = False

        # 将连接关系添加到edges中
        if is_connect:
            edges.append([cha1, cha2])
            x = [pos[cha1][0], pos[cha2][0]]
            y = [pos[cha1][1], pos[cha2][1]]
            plt.plot(x, y)

np.save(f'edges{channel_num}.npy', edges)
plt.show()
