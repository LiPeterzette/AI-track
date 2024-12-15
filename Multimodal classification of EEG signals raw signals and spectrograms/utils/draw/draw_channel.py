import math
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import matplotlib.cm as cm
dataset_name = 'KUL'
subject_num_dict = {'KUL': 16, 'DTU': 18, 'SCUT': 20}
subject_num = subject_num_dict[dataset_name]
test_all = False
sub_id = 10
# data1020 = pd.read_excel('pos.xlsx', index_col=0)
# channels1020 = np.array(data1020.index)
# pos = np.array(data1020)
pos = np.load('../pos.npy')
# def draw_connect(sub_id,dataset_name):

def upper_quartile(matrix):
    # 将矩阵转换为一维数组
    data = np.array(matrix).flatten()
    # 对数据进行排序
    data.sort()
    # 计算上四分位数的位置
    index = int(len(data) * 0.99)
    # 返回上四分位数
    return data[index]



def plot_connection(sub_id,dataset_name,test_all=False):
    if test_all:
        adj_matrix = []
        for k_sub in range(1,subject_num+1):
            adj = np.load(f'../../channel_score/{dataset_name}/adj_S{k_sub}_64_{dataset_name}.npy')
            # adj = np.mean(adj, axis=0)
            adj_matrix.append(adj)
        adj_matrix = np.array(adj_matrix)
        adj_matrix = np.mean(adj_matrix,axis=0)
    else:
        adj_matrix = np.load(f'../../channel_score/{dataset_name}/adj_S{sub_id}_64_{dataset_name}.npy')
        # adj_matrix = np.mean(adj_matrix,axis=0)

    for j in range(64):
        adj_matrix[j, j] = 0
    adj_matrix = adj_matrix+adj_matrix.transpose(1,0)
    D = np.mean(adj_matrix,axis=0)*50
    # norms = np.sqrt(np.sum(adj_matrix**2, axis=1, keepdims=True))
    #
    # # 对每一行进行归一化
    # adj_matrix = adj_matrix / norms

    # 将小于 0 或低于中位数的元素置零
    # median = np.median(adj_matrix)
    # adj_matrix[adj_matrix < 0] = 0
    # adj_matrix[adj_matrix < median] = 0

    # 保留前N大的边

    adj = adj_matrix
    arr_flat = adj.flatten()
    arr_flat_sorted = np.sort(arr_flat)[::-1]
    threshold = arr_flat_sorted[40]
    adj[adj < threshold] = 0
    adj = np.reshape(adj,(64,64))
    row, col = np.nonzero(adj)
    channel_select = np.concatenate((row, col), axis=None)
    channel_select = np.unique(channel_select)
    # 标出选出来的结点
    # adj = adj_matrix
    # channels_num = 16
    # channel_select = []
    # score = np.zeros([64])
    # for i in range(1, 100):
    #     i = np.argmax(adj)
    #     a, b = divmod(i, adj.shape[1])
    #     adj[a,b] = 0
    #     adj[b,a] = 0
    #     if a in channel_select:
    #         score[a] = score[a] + 1
    #     else:
    #         channel_select.append(a)
    #     if b in channel_select:
    #         score[b] = score[b] + 1
    #     else:
    #         channel_select.append(b)
    #     l = len(channel_select)
    #     if l >= channels_num+1:
    #         del (channel_select[-1])
    #     if l >= channels_num:
    #         break
    pos_select = pos[channel_select]
    D1 = D[channel_select]
    # for i in range(len(adj_matrix)):
    #     row = adj_matrix[i]
    #     row_sorted = np.sort(row)[::-1]
    #     threshold = row_sorted[1]
    #     adj_matrix[i][adj_matrix[i] < threshold] = 0
    # row, col = np.nonzero(adj_matrix)
    #

    linewidths = adj_matrix[row, col]

    norm = plt.Normalize(min(linewidths), max(linewidths))
    cmap = cm.ScalarMappable(norm=norm, cmap='plasma')

    # 获取线的颜色


    # 绘制连接线D
    fig, ax = plt.subplots(figsize=(1, 1), dpi=900)
    ax.scatter(pos[:, 0], pos[:, 1], c='SkyBlue', marker='o', s=D)  # 画出小实心圆点
    ax.scatter(pos_select[:, 0], pos_select[:, 1], c='Tomato', marker='o',s=D1)  # 画出小实心圆点


    for i in range(len(row)):
        color = cmap.to_rgba(linewidths[i])
        ax.plot(
                 [pos[col[i]][0], pos[row[i]][0]],[pos[col[i]][1], pos[row[i]][1]],
                 color="black", linewidth=0.05)


    # lines = []
    # for i in range(len(row)):
    #     lines.append([(pos[col[i]][0], pos[col[i]][1]), (pos[row[i]][0], pos[row[i]][1])])
    # lc = LineCollection(lines, linewidths=0.1, cmap='plasma')
    # # 绘制图形
    # ax.add_collection(lc)
    # ax.autoscale()
    # ax.margins(0.01)
    # 添加颜色条
    # cbar = plt.colorbar(lc,fraction=0.05)
    ax.axis('off')  # 不显示坐标轴
    ax.axis('equal')
    plt.savefig(f'C:/Users/zxz/Desktop/origin绘图/DTU/Connection/S{sub_id}_{dataset_name}.png',bbox_inches='tight')
    # plt.savefig(f'C:/Users/zxz/Desktop/origin绘图/连接图/S{sub_id}_{dataset_name}.png')
    plt.show()


if __name__ == '__main__':
    for i in range(1,17):
        plot_connection(sub_id=i,dataset_name="KUL")