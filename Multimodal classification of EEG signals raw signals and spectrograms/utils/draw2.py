import math
import mne
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


data1020 = pd.read_excel('pos.xlsx', index_col=0)
channels1020 = np.array(data1020.index)
pos = np.array(data1020)
distance_dict = {'64': 0.025, '32': 0.034, '16': 0.034}
channel_num = 64
# pos = np.load('pos.npy')
# channel_index = less_channel(channel_num)
plt.figure(dpi=1800)
plt.axis('equal')
plt.axis('off')

biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
channel64_names = biosemi64_montage.ch_names
biosemi_montage = mne.channels.make_standard_montage(f'biosemi{channel_num}')
channel_names = biosemi_montage.ch_names
# biosemi_montage.plot(show_names=True)
channel_index = [channel64_names.index(channel) for channel in channel_names]
pos = pos[channel_index, :]
# 绘制电极坐标
X_data, Y_data = [], []
for k_ch in range(channel_num):
    X_data.append(pos[k_ch][0])
    Y_data.append(pos[k_ch][1])
    plt.text(pos[k_ch][0], pos[k_ch][1], channel64_names[k_ch], fontsize=1, verticalalignment='center', color='black',
             horizontalalignment='center', zorder=8)
    # # 绘制电极圆圈
    circle = plt.Circle(pos[k_ch], 0.005, color='#86c166', fill=True, linewidth=0.2, zorder=2)
    plt.gcf().gca().add_artist(circle)

# draw(channel_num=64, zorder=2, color='red')

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
            if channel_names[cha1] == 'T7' and channel_names[cha2] in ['FT7', 'TP7']:
                is_connect = True
            if channel_names[cha1] == 'T8' and channel_names[cha2] in ['FT8', 'TP8']:
                is_connect = True
            # 删除边缘连接
            if channel_names[cha1] in ['P5', 'P6'] and channel_names[cha2] in ['P9', 'P10']:
                is_connect = False
            if channel_names[cha2] in ['P5', 'P6'] and channel_names[cha1] in ['P9', 'P10']:
                is_connect = False
            if channel_names[cha1] in ['F3', 'F5', 'F4', 'F6'] and channel_names[cha2] in ['AF7', 'AF8']:
                is_connect = False
            if channel_names[cha2] in ['F3', 'F5', 'F4', 'F6'] and channel_names[cha1] in ['AF7', 'AF8']:
                is_connect = False
            if channel_names[cha2] in ['AF3', 'AF4'] and channel_names[cha1] in ['F1', 'F2']:
                is_connect = False
            if channel_names[cha1] in ['AF3', 'AF4'] and channel_names[cha2] in ['F1', 'F2']:
                is_connect = False
            if channel_names[cha1] in ['F7'] and channel_names[cha2] in ['FC5']:
                is_connect = False
            if channel_names[cha1] in ['FC5'] and channel_names[cha2] in ['F7']:
                is_connect = False
            if channel_names[cha1] in ['F5'] and channel_names[cha2] in ['FC3']:
                is_connect = False
            if channel_names[cha2] in ['F5'] and channel_names[cha1] in ['FC3']:
                is_connect = False
            if channel_names[cha1] in ['F6'] and channel_names[cha2] in ['FC4']:
                is_connect = False
            if channel_names[cha2] in ['F6'] and channel_names[cha1] in ['FC4']:
                is_connect = False
            if channel_names[cha1] in ['F8'] and channel_names[cha2] in ['FC6']:
                is_connect = False
            if channel_names[cha2] in ['F8'] and channel_names[cha1] in ['FC6']:
                is_connect = False
            if channel_names[cha1] in ['F6'] and channel_names[cha2] in ['FC4']:
                is_connect = False
            if channel_names[cha2] in ['F6'] and channel_names[cha1] in ['FC4']:
                is_connect = False
            if channel_names[cha1] in ['P7'] and channel_names[cha2] in ['CP5']:
                is_connect = False
            if channel_names[cha2] in ['P7'] and channel_names[cha1] in ['CP5']:
                is_connect = False
            if channel_names[cha1] in ['P5'] and channel_names[cha2] in ['CP3']:
                is_connect = False
            if channel_names[cha2] in ['P5'] and channel_names[cha1] in ['CP3']:
                is_connect = False
            if channel_names[cha1] in ['P6'] and channel_names[cha2] in ['CP4']:
                is_connect = False
            if channel_names[cha2] in ['P6'] and channel_names[cha1] in ['CP4']:
                is_connect = False
            if channel_names[cha1] in ['P8'] and channel_names[cha2] in ['CP6']:
                is_connect = False
            if channel_names[cha2] in ['P8'] and channel_names[cha1] in ['CP6']:
                is_connect = False
            if channel_names[cha1] in ['PO7', 'PO3', 'PO4', 'PO8'] and channel_names[cha2] in ['P5', 'P3', 'P1', 'P2', 'P4', "P6"]:
                is_connect = False
            if channel_names[cha2] in ['PO7', 'PO3', 'PO4', 'PO8'] and channel_names[cha1] in ['P5', 'P3', 'P1', 'P2', 'P4', "P6"]:
                is_connect = False
            if channel_names[cha2] in ['PO3'] and channel_names[cha1] in ['P3']:
                is_connect = True
            if channel_names[cha2] in ['PO4'] and channel_names[cha1] in ['P4']:
                is_connect = True
        # 将连接关系添加到edges中
        if is_connect:
            edges.append([cha1, cha2])
            x = [pos[cha1][0], pos[cha2][0]]
            y = [pos[cha1][1], pos[cha2][1]]
            plt.plot(x, y, zorder=1, color='black', linewidth=0.08)

# np.save(f'edges{channel_num}.npy', edges)
plt.show()

