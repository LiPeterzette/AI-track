import torch
import torch.optim as optim
from collections import defaultdict
from preprocess import preprocess
from my_splite import data_split
from model import *
from sklearn import preprocessing
# from model.sag_pool_ import sagPool
from model.CNN import CNN
from model.AGGCN import AGGCN
from model.mymodel import Mymodel
from model.AGSL_net import AGSL_net
from model.graph_learn import Graph_learn
from model.TGCN import TGCN
from model.multi_grapg import Graph_learn3
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from train2 import train2
import xlwt
import mne
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchstat import stat
from thop import profile
from thop import clever_format

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', device)
time_len = 1
test_version = 'AGCN18A3'
dataset_name = 'KUL'
# 模型设置
graph_layer_num = 1
graph_convolution_kernel = 7
l_freq, h_freq = 1, 32
lr = 0.001
overlap = 0
is_ica = True
epochs = 120
is_cross_subject = True
subject_num_dict = {'KUL': 16, 'DTU': 18, 'SCUT': 20}
subject_num = subject_num_dict[dataset_name]
sample_len,  channels_num = int(128 * time_len), 64
nets = defaultdict()
batch_dirt = {0.1: 256, 0.2: 256, 0.5: 64, 1: 32, 2: 16, 5: 8, 10: 4}
batch_size = batch_dirt[time_len] if not is_cross_subject else 512
nets['AGGCN'] = AGGCN
nets['AGSL_net'] = AGSL_net
nets['CNN'] = CNN
nets['mymodel'] = Mymodel
nets['graph_learn'] = Graph_learn
nets['TGCN'] = TGCN
nets['multi_grapg'] = Graph_learn3


def saveMatrix2Excel(data, path):
    # 将np的矩阵数组保存至表格
    data =data.T
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, float(data[i, j]))
    f.save(path)

def scale_data(data):
    for i in range(data.shape[0]):
        data[i, ...] = preprocessing.scale(data[i, ...])

def set_random_seed(seed):
        """Set random seed.
        Args:
            seed (int): Seed to be used.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Default: False.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)#cpu
        torch.cuda.manual_seed_all(seed)#gpu

def less_channel():

    biosemi64_montage = mne.channels.make_standard_montage('biosemi64')
    channel64_names = biosemi64_montage.ch_names
    biosemi_montage = mne.channels.make_standard_montage(f'biosemi{channels_num}')
    channel_names = biosemi_montage.ch_names
    channel_index = [channel64_names.index(channel) for channel in channel_names]
    return channel_index

def generate_data_info(subject_id, sorted_=True, edge_type='my', feature_type='psd_group', net_name='CNN'):
    channel_index = less_channel()
    x_train = np.zeros((0, sample_len, channels_num))
    y_train = np.zeros((0,))
    x_test = np.zeros((0, sample_len, channels_num))
    y_test = np.zeros((0,))
    # 加载所有人的数据
    train_subject_num = subject_num_dict["KUL"]
    for k_sub in range(train_subject_num):
        print('sub:' + str(k_sub))
        eeg, voice, label = preprocess("KUL", k_sub+1, l_freq=l_freq, h_freq=h_freq, is_ica=True)
        eeg, voice, label = data_split(eeg, voice, label, 1, time_len, overlap)
        data = eeg
        data = data[:, :, channel_index]
        x_train = np.concatenate((x_train, data), axis=0)
        y_train = np.concatenate((y_train, label), axis=0)
    test_subject_num = subject_num_dict["DTU"]
    for i_sub in range(subject_num):
        print('sub:' + str(i_sub))
        eeg, voice, label = preprocess("DTU", i_sub+1, l_freq=l_freq, h_freq=h_freq, is_ica=True)
        eeg, voice, label = data_split(eeg, voice, label, 1, time_len, overlap)
        data = eeg
        data = data[:, :, channel_index]
        x_test = np.concatenate((x_test, data), axis=0)
        y_test = np.concatenate((y_test, label), axis=0)
    index1 = [i for i in range(x_train.shape[0])]
    random.shuffle(index1)
    x_train = x_train[index1]
    y_train = y_train[index1]
    # 打乱数据（测试集）
    index2 = [i for i in range(x_test.shape[0])]
    random.shuffle(index2)
    x_test = x_test[index2]
    y_test = y_test[index2]
    data_info = defaultdict()
    data_info['x_train'], data_info['x_test'] = x_train, x_test
    data_info['y_train'], data_info['y_test'] = y_train, y_test
    return data_info


class Dataset(TensorDataset):
    def __init__(self, data, lable):
        super(Dataset, self).__init__()
        self.samples_data = data.tolist()
        self.samples_label = lable.tolist()
        self.num_samples = len(self.samples_data)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, sample_idx):
        data = self.samples_data[sample_idx]
        label = self.samples_label[sample_idx]
        data_tensor = torch.Tensor(data)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return data_tensor, label_tensor

def model_fit(data_info,x_train,y_train):
    nets = data_info['nets']
    batch_size = data_info['batch_size']
    epochs = data_info['epochs']
    train_num = x_train.shape[0]
    print(f'Processed EEG train data shape: {x_train.shape}')
    # print(f'Processed EEG test data shape: {x_test.shape}')
    train_list = Dataset(x_train,y_train)
    # test_list = Dataset(x_test,y_test)
    num_features = x_train.shape[1]
    train_loader = DataLoader(train_list,
                                  batch_size=batch_size,shuffle=True)
    model = nets[net_name](num_features, kernel=graph_convolution_kernel,channel_num=channels_num).to(device)
    criterion = torch.nn.BCELoss()
    if model == AGSL_net:
        small_lr_layers = list(map(id, model.weight_key, model.weight_query))
        optimizer = optim.Adam([
            {"params":small_lr_layers, "lr": lr}
            ], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    patience = 10 if is_cross_subject else 50
    acc = train2(model, criterion, optimizer, train_loader, device, train_num, epochs,
                 patience=patience)
    acc = acc.item()
    del model
    return acc

def main(nets, net_name, data_info, batch_size, epochs):
    accuracies = []
    data_info['nets'] = nets
    data_info['net_name'] = net_name
    data_info['batch_size'] = batch_size
    data_info['epochs'] = epochs
    if is_cross_subject:
        x_train, x_test = data_info['x_train'], data_info['x_test']
        y_train, y_test = data_info['y_train'], data_info['y_test']
        acc = model_fit(data_info, x_train, x_test, y_train, y_test)
        accuracies.append(acc)
        print(accuracies)
        print()
    else:
        data = data_info['data']
        label = data_info['label']
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=3)
        for train_index, test_index in sss.split(data, label):
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            acc = model_fit(data_info, x_train, x_test, y_train, y_test)
            accuracies.append(acc)
            print(accuracies)
            print()
    print(np.mean(accuracies))
    print(accuracies)
    return accuracies

if __name__ == '__main__':
    """
    hyper-parameter search for multi-class data
    """
    seed = 20
    subject_id = 1
    edge_type = 'my'
    feature_type = 'my_fea'
    net_names = ["AGSL_net",'mymodel']
    batch_size = 32
    epochs = epochs
    _sorted = True
    test_all = False

    num_model = len(net_names)
    accs = np.zeros([num_model, 16, 5])
    if test_all:
        i = 0
        for net_name in net_names:
            print(net_name)
            for subject_id in range(16):
                data_info = generate_data_info(subject_id+1, _sorted, edge_type, feature_type, net_name, is_cross_subject)
                # set_random_seed(seed)
                logged = False
                # logged = True
                acc = main(nets, net_name, data_info, batch_size, epochs)
                # acc = [1,1,1,1,1]
                accs[i, subject_id, :] = acc
                print(subject_id+1)
                print(net_name)
                print(accs)
            i = i + 1
    else:
        i = 0
        for net_name in net_names:
            data_info = generate_data_info(subject_id, _sorted, edge_type, feature_type, net_name, is_cross_subject)
            # set_random_seed(seed)
            logged = False
            acc = main(nets, net_name, data_info, batch_size, epochs)
            print(net_name)
            accs[i, :] = acc
            print(subject_id)
            i = i + 1
            print(accs)
    for t in range(accs.shape[0]):
        accs1 = np.array(accs[t])
        accs1 = np.transpose(accs1, [1, 0])
        path = f"result/{net_names[t]}_{time_len}S_{dataset_name}.xls"  # 保存在当前文件夹下，你也可以指定绝对路径
        saveMatrix2Excel(accs1, path)