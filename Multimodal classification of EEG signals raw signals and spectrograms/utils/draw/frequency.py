import numpy as np
from matplotlib import pyplot as plt
from energy_plot import plot_eeg_topomap,channel_names_KUL
import xlwt

def count_numbers(lst):
    score = [0] * 64
    Distribution = [0] * 64
    for num in lst:
        if 0 <= num <= 63:
            score[num] += 1
    for i in range(64):
        Distribution[i] = score[i] / len(lst)
    return score, Distribution

import matplotlib.pyplot as plt
def draw_bar_chart(data, names):
    plt.figure(figsize=(20, 8))
    if len(data) != 64 or len(names) != 64:
        raise ValueError("Both data and names lists should have a length of 64.")

    # plt.bar(range(len(data)), data)
    # plt.xticks(range(len(names)), names, rotation=45,fontsize=16)
    # plt.ylabel("Frequency", fontsize=20)
    # plt.yticks(fontsize=16)
    # plt.xlabel("Channel", fontsize=20)

    # 对数据和名称进行排序
    sorted_data, sorted_names = zip(*sorted(zip(data, names), reverse=True))
    saveMatrix2Excel(sorted_names, "name.xls")
    M = np.array(range(62))
    # 只绘制前16个最大值的柱状图
    plt.bar(M, sorted_data[:62])
    plt.yticks(fontsize=14)
    plt.xticks(M, sorted_names[:62], rotation=90,fontsize=14,fontdict={'family' : 'Times New Roman', 'size': 17})
    plt.xlabel("Channel", fontsize=20,fontdict={'family' : 'Times New Roman', 'size': 20})
    plt.ylabel("Frequency", fontsize=20,fontdict={'family' : 'Times New Roman', 'size': 20})

    plt.savefig('C:/Users/zxz/Desktop/origin绘图/DTU/Connection/Frequency.png', dpi=750, bbox_inches='tight')
    plt.show()


subject_num = 16

def saveMatrix2Excel(data, path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    # 将np的矩阵数组保存至表格
    for i in range(64):
        sheet1.write(i, 1, str(data[i]))
        f.save(path)

def calculate_cha_fre(dataset_name):
    pos = np.load('../pos.npy')
    channel_selects = []
    D_ = []
    for k_sub in range(1, subject_num + 1):
        adj_matrix = np.load(f'../../channel_score/{dataset_name}/adj_S{k_sub}_64_{dataset_name}.npy')
        for j in range(64):
            adj_matrix[j, j] = 0
        adj_matrix = adj_matrix + adj_matrix.transpose(1, 0)
        D = np.mean(adj_matrix, axis=0) * 50
        # norms = np.sqrt(np.sum(adj_matrix**2, axis=1, keepdims=True))
        #
        # # 对每一行进行归一化
        # adj_matrix = adj_matrix / norms

        # 将小于 0 或低于中位数的元素置零
        # median = np.median(adj_matrix)
        # adj_matrix[adj_matrix < 0] = 0
        # adj_matrix[adj_matrix < median] = 0

        # 保留前N大的边
        D_.append(D)
        adj = adj_matrix
        arr_flat = adj.flatten()
        arr_flat_sorted = np.sort(arr_flat)[::-1]
        threshold = arr_flat_sorted[32]
        adj[adj < threshold] = 0
        adj = np.reshape(adj, (64, 64))
        row, col = np.nonzero(adj)
        channel_select = np.concatenate((row, col), axis=None).tolist()
        channel_selects.extend(channel_select)
    D_ = np.array(D_)
    D_ = np.mean(D_,axis=0)
    frequency, Distribution = count_numbers(channel_selects)
    print("Score:", frequency)
    print("Distribution:", frequency)
    draw_bar_chart(frequency, channel_names_KUL)
    # plot_eeg_topomap(energy_matrix=D_,pos=pos,filename="D.png")
    # plot_eeg_topomap(energy_matrix=Distribution, pos=pos,filename="Distribution.png")

if __name__ == '__main__':
    calculate_cha_fre(dataset_name="KUL")



