import matplotlib.pyplot as plt
import numpy as np
import mne
from preprocess import preprocess,data_loader
from my_splite import data_split
 # 通道名称
channel_num = 9
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                      'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10',
                      'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4',
                      'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
                      'FT9', 'FT10', 'Fpz', 'CPz', 'FCz']

raw_data, voice, label = data_loader('KUL', 7)
raw_data, voice, label = data_split(raw_data, voice, label, 1, 20, 0)
is_black =False
ch_names = ch_names[:channel_num]
ch_types = ['eeg'] * len(ch_names)  # 通道类型
sfreq = 100  # 采样率
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_data = np.array(raw_data[50,400:2000,:channel_num])
data = raw_data.transpose(1,0)

if is_black:
    raw_data = data*0.01
    raw = mne.io.RawArray(data, info)
    # raw.plot(n_channels=len(ch_names), scalings=dict(eeg=50))
    start, stop = 10.0, 12.0
    raw.plot(n_channels=len(ch_names), start=start, duration=stop-start, scalings={'eeg': 30})
    plt.show()
else:
    # 取各通道信号的最大值和最小值
    max_vals = np.max(data, axis=1)
    min_vals = np.min(data, axis=1)
    # 计算通道最大值和最小值的差值，并将差值相加得到总差值
    total_diff = np.sum(max_vals - min_vals)

    # 计算平均偏移量
    mean_offset = total_diff / (data.shape[0]+200)

    # 计算各个通道曲线的 y 轴偏移量
    offsets = np.zeros_like(max_vals)
    offsets[0] = 0
    for i in range(1, data.shape[0]):
        offsets[i] = offsets[i-1] + max_vals[i-1] - min_vals[i] + mean_offset

    # 绘制时序信号
    plt.figure(figsize=(8, 6))
    for i in range(data.shape[0]):
        y = data[i] + offsets[i]
        plt.plot(y)

    # 设置图形属性
    plt.title('Time Series Signal')
    plt.xlabel('Time')
    plt.ylabel('Channel')
    plt.xlim(0, data.shape[1]-1)
    plt.ylim(np.min(data) + np.min(offsets), np.max(data) + np.max(offsets))
    plt.legend()
    plt.show()






