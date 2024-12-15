import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, np.pi, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.sin(Y)

# 绘制颜色图
fig, ax = plt.subplots()
im = ax.pcolormesh(x, y, Z)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.05)
cbar.ax.yaxis.set_ticks_position("right")
cbar.ax.yaxis.set_label_position("right")
cbar.set_label("Z")

# 设置颜色条的大小和位置
fig.subplots_adjust(right=0.8)
ax_new = fig.add_axes([0.85, 0.15, 0.05, 0.7])

# 让 ax_new 坐标轴对象与原始坐标轴对齐，并绘制另一个图形
ax_new.spines["top"].set_visible(False)
ax_new.spines["right"].set_visible(False)
ax_new.spines["bottom"].set_visible(False)
ax_new.set_xticks([])
ax_new.set_yticks([])
ax_new.set_xlim([0, 1])
ax_new.set_ylim([0, 1])
ax_new.imshow(np.array([[1, 1], [0, 0]]), cmap=plt.cm.Blues)

plt.show()