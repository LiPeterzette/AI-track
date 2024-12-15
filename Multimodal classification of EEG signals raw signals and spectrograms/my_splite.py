# python3
# encoding: utf-8
# 
# @Time    : 2022/05/13 10:24
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : my_splite.py
# @Software: Pycharm
import numpy as np

from preprocess import preprocess


def data_split(eeg, voice, label, target, time_len, overlap):
    """
    将序列数据转化为样本，并按照5折的方式的排列整齐
    不兼容128Hz以外的数据（含有语音和脑电信号）
    :param eeg: 脑电数据，列表形式的Trail，每个Trail的形状是Time*Channels
    :param voice: 语音数据，列表形式的Trail，每个Trail的形状是Time*2
    :param label: 标签数据，根据targe决定输出标签
    :param target: 确定左右/讲话者的分类模型
    :param time_len: 样本的长度
    :param overlap: 样本的重叠率
    :return:
    """

    sample_len = int(128 * time_len)

    my_trail_samples = np.empty((5, 0, sample_len, eeg[0].shape[-1]))
    my_voice_samples = np.empty((5, 0, sample_len, 2))
    my_labels = np.empty((5, 0))

    for k_tra in range(len(eeg)):
        trail_eeg = eeg[k_tra]
        trail_voice = voice[k_tra]
        trail_label = label[k_tra][target]
        trail_voice = np.transpose(trail_voice, axes=[1,0])
        # 确定重叠率的数据
        over_samples = int(sample_len * (1 - overlap))
        over_start = list(range(0, sample_len, over_samples))
        # 根据起点划分数据
        for k_sta in over_start:
            tmp_eeg = set_samples(trail_eeg, k_sta, sample_len, overlap)
            tmp_voice = set_samples(trail_voice, k_sta, sample_len, overlap)

            my_trail_samples = np.concatenate((my_trail_samples, tmp_eeg), axis=1)
            # my_voice_samples = np.concatenate((my_voice_samples, tmp_voice), axis=1)
            my_labels = np.concatenate((my_labels, trail_label * np.ones((5, tmp_eeg.shape[1]))), axis=1)

    # 转化为单一维度的数据
    my_trail_samples = np.reshape(my_trail_samples, [-1, my_trail_samples.shape[2], my_trail_samples.shape[3]])
    # my_voice_samples = np.reshape(my_voice_samples, [-1, my_voice_samples.shape[2], my_voice_samples.shape[3]])
    my_labels = np.reshape(my_labels, -1)

    return my_trail_samples, [], my_labels


def set_samples(trail_data, k_sta, sample_len, overlap):
    # 切分整数长度
    data_len, channels_num = trail_data.shape[0], trail_data.shape[1]
    k_end = (data_len - k_sta) // sample_len * sample_len + k_sta
    trail_data = trail_data[k_sta:k_end, :]

    # cutoff
    trail_data = np.reshape(trail_data, [-1, sample_len, channels_num])

    # 划分为5折数据
    # TODO: 检查数据是否为连续时间序列，方便后续的五折交叉验证
    trails_num = trail_data.shape[0] // 5 * 5
    trail_data = trail_data[0:trails_num, :, :]
    trail_data = np.reshape(trail_data, [5, int(trail_data.shape[0] / 5), sample_len, channels_num])

    if overlap != 0:
        trail_data = trail_data[:, 1:-1, :]

    return trail_data


if __name__ == '__main__':
    # # # 构造测试数据，包含eeg、voice、label
    # eeg = [np.random.random((129 * 50, 64))]
    # voice = [np.random.random((128 * 50, 2))]
    # label = [[0, 1]]
    # # set_samples(eeg, 64, 128, 0.5)
    eeg, voice, label = preprocess('KUL', '2', 1, 32)
    eeg, voice, label = data_split(eeg, voice, label, 1, 1, 0.5)
    print(eeg.shape)
    print(voice.shape)
