import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import EvokedArray,create_info


channel_names_KUL = ['Fp1',  'AF7',  'AF3',  'F1',  'F3',  'F5',  'F7',  'FT7',  'FC5',  'FC3',  'FC1',  'C1'
    ,  'C3',  'C5',  'T7',  'TP7',  'CP5',  'CP3',  'CP1',  'P1',  'P3',  'P5',  'P7',  'P9',  'PO7',  'PO3',  'O1',
                     'Iz',  'Oz',  'POz',  'Pz',  'CPz',  'Fpz',  'Fp2',  'AF8',  'AF4',  'AFz',  'Fz',  'F2',  'F4',  'F6',
                     'F8',  'FT8',  'FC6',  'FC4',  'FC2',  'FCz',  'Cz',  'C2',  'C4',  'C6',  'T8',  'TP8',  'CP6',  'CP4',
                     'CP2',  'P2',  'P4',  'P6',  'P8',  'P10',  'PO8',  'PO4',  'O2']

# def plot_eeg_topomap(energy_matrix):
#     # 创建虚拟的脑电响应对象
#     info = mne.create_info(ch_names=channel_names, sfreq=1, ch_types='eeg')
#     evoked = mne.EvokedArray(energy_matrix, info)
#     # 绘制脑电地形图
#     evoked.plot_topomap(times=0, unit='a.u.', time_format='')


def plot_eeg_topomap(energy_matrix, pos,filename):
    """
    使用MNE库根据脑电信号通道的能量值矩阵绘制脑电地形图。

    参数：
    energy_matrix (numpy.ndarray): 能量值矩阵，每一行表示一个通道的能量值。
    channel_names (list of str): 通道名称列表。

    返回：
    None
    """
    vmin = np.min(energy_matrix)
    vmax = np.max(energy_matrix)
    fig, ax = plt.subplots()
    im,cm = mne.viz.plot_topomap(energy_matrix,pos,show=False,cmap="viridis",axes=ax)
    # cbar = plt.colorbar(ax.collections[0], ax=ax,fraction=0.05, cmap="viridis")
    # cbar.set_clim(vmax=vmax,vmin=vmin)
    ax_x_start = 0.85
    ax_x_width = 0.03
    ax_y_start = 0.15
    ax_y_height = 0.7
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    plt.savefig('C:/Users/zxz/Desktop/origin绘图/DTU/Connection/'+filename, dpi = 750, bbox_inches='tight')
    plt.show()
    # cbar.set_label('Data Value')
# def plot_eeg_topomap(energy_matrix,pos):
#     # 创建一个虚拟的脑电图信息对象
#     info = create_info(ch_names=channel_names_KUL, sfreq=1, ch_types='eeg')
#
#     # 将能量值矩阵转换为Evoked对象
#     evoked = EvokedArray(energy_matrix, info)
#
#     # 绘制脑电地形图
#     fig, ax = plt.subplots()
#     mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=ax, show_names=False, cmap='RdBu_r', contours=0)
#
#     # 添加颜色条
#     cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.5, pad=0.2)
#     cbar.set_label('Energy')
#
#     # 显示图像
#     plt.show()


if __name__ == '__main__':
    # 示例：使用随机生成的能量值矩阵
    pos = np.load('../pos.npy')
    energy_matrix = np.random.rand(64,1)
    energy_matrix = np.squeeze(energy_matrix)
    plot_eeg_topomap(energy_matrix,pos,filename="example.png")
    plt.show()


