import matplotlib.pyplot as plt
import numpy as np
from preprocess import normalization
from matplotlib.pyplot import MultipleLocator
from energy_plot import channel_names_KUL


dataset_name = 'KUL'
subject_num_dict = {'KUL': 16, 'DTU': 18, 'SCUT': 20}
subject_num = subject_num_dict[dataset_name]
test_all = True
sub_id = 10


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



# adj_matrix = adj_matrix+adj_matrix.transpose(1,0)
# adj = np.load('../channel_score/KUL_conv/adj_S10_64_KUL.npy')
adj = adj_matrix
# for j in range(64):
#     adj[j, j] = 1
# plt.matshow(adj, cmap=plt.get_cmap('BuGn'))  # , alpha=0.3
# plt.imshow(adj, cmap="OrRd")
plt.pcolormesh(adj,cmap="OrRd")
# plt.imshow(adj)
plt.rcParams['font.size'] = 15
# # 创建x轴定位器，间隔2
# x_major_locator = MultipleLocator(16)
# # 创建y轴定位器，间隔5
# y_major_locator = MultipleLocator(16)
#
# # # 获取轴对象
# # plt.axis('equal')
ax = plt.gca()
# # # 设置x轴的间隔
# ax.xaxis.set_major_locator(x_major_locator)
# # 设置y轴的间隔
# ax.yaxis.set_major_locator(y_major_locator)


names = channel_names_KUL
plt.yticks(range(len(names)), names, rotation=0,fontsize=5)
plt.xticks(range(len(names)), names, rotation=90,fontsize=5)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
ax.set_aspect('equal')
plt.colorbar(fraction=0.05, pad=0.01)

plt.show()
