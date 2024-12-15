import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
img1 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/merge_KUL.png')
img2 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/ALL_KUL.png')
img3 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/merge_DTU.png')
img4 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/ALL_DTU.png')

# fig = plt.figure(figsize=(12, 8),dpi=600)
# gs = fig.add_gridspec(2, 2, height_ratios=[10, 7])
#
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.set_title('(a)')
# ax1.imshow(img1)
# ax1.axis('off')
#
# # 在1行2列的第2个位置放置子图
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.imshow(img2)
# ax2.axis('off')
#
# # 在2行2列的第1个位置放置子图
# ax3 = fig.add_subplot(2,2, 3)
# ax3.imshow(img3)
# ax3.axis('off')
#
# # 在2行2列的第2个位置放置子图
# ax4 = fig.add_subplot(2,2, 4)
# ax4.imshow(img4)
# ax4.axis('off')
#
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('C:/Users/zxz/Desktop/origin绘图/连接图/merge_ALL', bbox_inches='tight')
# # 显示画布
# plt.show()
# 保存画布

# fig, ax = plt.subplots(2, 2, figsize=(80, 8), gridspec_kw={'height_ratios': [1, 1]})
#
# # 在每个子图中绘制对应的图片
# ax[0, 0].imshow(np.array(img1), aspect='equal')
# ax[0, 1].imshow(np.array(img2), aspect='equal')
# ax[1, 0].imshow(np.array(img3), aspect='equal')
# ax[1, 1].imshow(np.array(img4), aspect='equal')
#
# # 设置每张子图的标号'（a）','（b）'
# # ax[0, 0].text(0.05, -0.2, '（a）', transform=ax[0, 0].transAxes,
# #                fontsize=12, fontweight='bold', color='red')
# # ax[0, 1].text(0.05, -0.2, '（b）', transform=ax[0, 1].transAxes,
# #                fontsize=12, fontweight='bold', color='red')
# # ax[1, 0].text(0.05, -0.2, '（c）', transform=ax[1, 0].transAxes,
# #                fontsize=12, fontweight='bold', color='red')
# # ax[1, 1].text(0.05, -0.2, '（d）', transform=ax[1, 1].transAxes,
# #                fontsize=12, fontweight='bold', color='red')
#
# # 隐藏坐标轴
# for axis in ax.flatten():
#     axis.axis('off')
#
# # 指定图片间距和整体外边框
# plt.subplots_adjust(wspace=0, hspace=0)
#
# #
# # # 显示子图
# plt.show()




# 读取图片


# 计算高度比例


h1, w1, _ = img1.shape
h2, w2, _ = img2.shape
h_max = max(h1, h2)
height_ratio = [h1/h_max, (h2/h_max)*1.6]
w_sum = w1 + w2
width_ratio = [w1/w_sum, (w2/w_sum)*1.6]
# 创建子图布局
fig = plt.figure(figsize=(8, 2),dpi=300)
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=height_ratio, width_ratios=width_ratio, wspace=0.05, hspace=0)

# 调整子图大小，保持图片的比例不变
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img1)
ax1.set_title('Image 4')
ax1.axis('off')
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img2)
ax2.axis('off')
plt.savefig('C:/Users/zxz/Desktop/origin绘图/连接图/merge_ALL1', bbox_inches='tight')
plt.show()

