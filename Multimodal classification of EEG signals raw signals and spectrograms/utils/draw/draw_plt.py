

import matplotlib.pyplot as plt
import numpy as np

fig1_1 = np.array([48.84, 50. , 50.14, 50.52, 50.58, 50.43, 50.43])
fig1_2 = np.array([53.3828, 54.8498, 56.0235, 57.4042, 57.853, 58.3535, 57.525])
fig1_3 = np.array([49.85, 50.61, 50.84, 50.96, 51.06, 51.21, 51.52])
fig2_1 = np.array([1., 39.25, 46.07, 50.54, 50.58, 50.22, 50.08, 48.7, 46.49, 4.21])
fig2_2 = np.array([0.5178, 7.525, 34.622, 37.2454, 53.3656, 56.0235, 56.8174, 57.7667, 57.387, 45.1329])
fig2_3 = np.array([1., 40.82, 42.74, 50.29, 50.8, 51.01, 50.73, 50.23, 48.45, 37.17])
fig3_1 = np.array([44.35, 47.73, 50.32, 51.7, 51.93, 51.92, 51.56, 50.32])
fig3_2 = np.array([38.9196, 41.3359, 51.7087, 55.5747, 56.8174, 55.989, 56.4895, 55.9544])
fig3_3 = np.array([44.38, 48.42, 50.96, 51.36, 51.47, 51.35, 50.86, 50.36])

x1 = np.array([1,2,3,5,8,10,12])

x2 = np.array([0.1,1,2,5,8,10,12,15,20,100])
x3 = np.array([1,50,100,150,200,250,300,350])
x2 = list(x2)
x3 = list(x3)
x2[0] = str('β:0.1')
x3[0] = str('C=1')
# 创建3个子图，每个子图包含一个数据集的三个版本
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

markers = ['o', 's', '^']
labels = ['Fig 1', 'Fig 2', 'Fig 3']
colors = ['darkorange', 'mediumorchid', 'forestgreen']
# colors = ['darkorange', 'mediumorchid', 'forestgreen']
# 绘制第一个子图
for i, fig_data in enumerate([fig1_1, fig1_2, fig1_3]):
    axes[0].plot(range(1, len(fig_data) + 1), fig_data, marker=markers[i], color=colors[i], label=labels[0] + f'_{i+1}')

axes[0].legend(loc='upper left', fontsize=12)
# axes[0].set_title('Data Set 1', fontsize=16)
axes[0].set_xlabel('(a)XX Effect of GKE parameters', fontsize=14)
axes[0].set_ylabel('Avg.Acc at last phase(%)', fontsize=14)
x_tick1 = range(1, len(x1) + 1)
axes[0].set_xticks(x_tick1)
axes[0].set_xticklabels([str(item)+'k' for item in x1])

axes[0].grid()
data1 = np.array((fig1_1, fig1_2, fig1_3))
a1 = np.max(data1)
b1 = np.min(data1)
axes[0].set_xlim([0, 8])
axes[0].set_ylim([b1-1, a1+1])
# axes[0].set_aspect(aspect=0.5)
# 绘制第二个子图
for i, fig_data in enumerate([fig2_1, fig2_2, fig2_3]):
    axes[1].plot(range(1, len(fig_data) + 1), fig_data, marker=markers[i], color=colors[i], label=labels[1] + f'_{i+1}')
axes[1].legend(loc='upper left', fontsize=12)
# axes[1].set_title('Data Set 2', fontsize=16)
axes[1].set_xlabel('(b)XX Effect of width parameters', fontsize=14)
# axes[1].set_ylabel('Avg.Acc at last phase', fontsize=14)
x_tick2 = range(1, len(x2) + 1)
axes[1].set_xticks(x_tick2)
axes[1].set_xticklabels(x2)
axes[1].grid()

data2 = np.array((fig2_1, fig2_2, fig2_3))
a2 = np.max(data2)
b2 = np.min(data2)
axes[1].set_xlim([0, 11])
axes[1].set_ylim([b2-1, a2+1])
# axes[1].set_aspect(aspect=0.2)
# 绘制第三个子图
for i, fig_data in enumerate([fig3_1, fig3_2, fig3_3]):
    axes[2].plot(range(1, len(fig_data) + 1), fig_data, marker=markers[i], color=colors[i], label=labels[2] + f'_{i+1}')
axes[2].legend(loc='upper left', fontsize=12)
# axes[2].set_title('Data Set 3', fontsize=16)
axes[2].set_xlabel('(b)Effect of AFC parameters', fontsize=14)
# axes[2].set_ylabel('Avg.Acc at last phase(%)', fontsize=14)
x_tick3 = range(1, len(x3) + 1)
axes[2].set_xticks(x_tick3)
axes[2].set_xticklabels(x3)
axes[2].grid()
data3 = np.array((fig3_1, fig3_2, fig3_3))
a3 = np.max(data3)
b3 = np.min(data3)
axes[2].set_xlim([0, 9])
axes[2].set_ylim([b3-1, a3+1])
# axes[2].set_aspect(aspect=0.5)
# 在绘制窗口中显示图形
plt.savefig("class1_xianzhang.pdf")
plt.show()
