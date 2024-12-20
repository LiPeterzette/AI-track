from tqdm import trange
from torch_geometric.data import Data, DataLoader
from utils.draw.edges import *
import torch
from .features import gen_features_hvg, gen_features_cre, gen_features_cre_group, gen_features_psd_group, gen_features_wavelet, gen_features_raw, gen_features_wt_deg, my_fea


def gen_data_list(data, label, edge_type='my', feature_type='psd_group', net = None):
    """
    Generate graph data list from matrix data and label.
    :param data: training or testing data in matrix form, shape: (N, T, C)
    :param label: training or testing label in matrix form, shape: (N, )
    :return: training or testing data list,
             each item in this list is a torch_geometric.data.Data object.
    """
    data_list = []
    for trial in trange(data.shape[0]):
        trial_data = data[trial, ...]
        trial_label = label[trial]

        # generate edge index and node features
        if edge_type == 'corr':
            edge_index, edge_weight = gen_edges_corr(trial_data)
        elif edge_type == 'wpli':
            edge_index, edge_weight = gen_edges_wpli(trial_data)
        elif edge_type == 'plv':
            edge_index, edge_weight = gen_edges_plv(trial_data)
        elif edge_type == 'cg':
            edge_index = gen_edges_cg(trial_data)
            edge_weight = np.zeros((edge_index.shape[-1], 1))
        elif edge_type == 'my':
            edge_index, edge_weight = gen_my_edge(trial_data)
        elif edge_type == 'direct':
            edge_index, edge_weight = dirictedge(trial_data)
        edge_index = torch.from_numpy(edge_index).long()
        edge_weight = torch.from_numpy(edge_weight).float()

        if net == 'multi_graph':
            x = np.transpose(trial_data, (0, 2, 1))
            x = torch.from_numpy(x).float()
            graph_data =MyData(X=x, edge_index=edge_index,
                              y=trial_label, edge_attr=edge_weight)
            data_list.append(graph_data)
        else:
            x = np.transpose(trial_data, (1, 0))
            x = torch.from_numpy(x).float()
            graph_data = Data(x=x, edge_index=edge_index,
                              y=trial_label, edge_attr=edge_weight)
            data_list.append(graph_data)
    return data_list

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'X'or'x':
            return None
        else:
          return super().__cat_dim__(key, value, *args, **kwargs)


def gen_dataloader(data_list, batch_size=32, shuffle=True):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def gen_Mydata_list(data, label, edge_type='my', feature_type='my_fee'):
    """
    Generate graph data list from matrix data and label.
    :param data: training or testing data in matrix form, shape: (N, T, C)
    :param label: training or testing label in matrix form, shape: (N, )
    :return: training or testing data list,
             each item in this list is a torch_geometric.data.Data object.
    """
    data_list = []
    for trial in trange(data.shape[0]):
        trial_data = data[trial, ...]
        trial_label = label[trial]

        # generate edge index and node features
        if edge_type == 'corr':
            edge_index, edge_weight = gen_edges_corr(trial_data)
        elif edge_type == 'wpli':
            edge_index, edge_weight = gen_edges_wpli(trial_data)
        elif edge_type == 'plv':
            edge_index, edge_weight = gen_edges_plv(trial_data)
        elif edge_type == 'cg':
            edge_index = gen_edges_cg(trial_data)
            edge_weight = np.zeros((edge_index.shape[-1], 1))
        elif edge_type == 'my':
            edge_index, edge_weight = gen_my_edge(trial_data)

        if feature_type == 'hvg':
            x = gen_features_hvg(trial_data)
        elif feature_type == 'cre':
            x = gen_features_cre(trial_data)
        elif feature_type == 'cre_group':
            x = gen_features_cre_group(trial_data)
        elif feature_type == 'psd_group':
            x = gen_features_psd_group(trial_data)
        elif feature_type == 'wavelet':
            x = gen_features_wavelet(trial_data, wavelet='coif1', level=4)
        elif feature_type == 'raw':
            x = gen_features_raw(trial_data)
        elif feature_type == 'wt_deg':
            x = gen_features_wt_deg(trial_data, level=5)
        elif feature_type == 'my_fea':
            x = my_fea(trial_data)


        edge_index = torch.from_numpy(edge_index).long()
        edge_weight = torch.from_numpy(edge_weight).float()
        x = torch.from_numpy(x).float()
        graph_data = MyData(X=x, edge_index=edge_index, y=trial_label, edge_attr=edge_weight)
        data_list.append(graph_data)
    return data_list

if __name__ == '__main__':
    pass
