# import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

import matplotlib.pyplot as plt
import numpy as np

# 生成4张样本图片
# dataset_name = 'KUL'
# img1 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S2_{dataset_name}.png')
# img2 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S3_{dataset_name}.png')
# img3 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S7_{dataset_name}.png')
# img4 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S8_{dataset_name}.png')
# img5 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S9_{dataset_name}.png')
# img6 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S12_{dataset_name}.png')
# img7= mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S13_{dataset_name}.png')
# img8 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S15_{dataset_name}.png')

dataset_name = 'DTU'
img1 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S2_{dataset_name}.png')
img2 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S3_{dataset_name}.png')
img3 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S8_{dataset_name}.png')
img4 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S9_{dataset_name}.png')
img5 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S10_{dataset_name}.png')
img6 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S11_{dataset_name}.png')
img7= mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S13_{dataset_name}.png')
img8 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/S14_{dataset_name}.png')


# 构造子图布局
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,16))
#
# # 在第1个子图中显示图像1
# ax1.imshow(img1)
# ax1.axis('off')
#
# # 在第2个子图中显示图像2
# ax2.imshow(img1)
# ax2.axis('off')
#
# # 在第3个子图中显示图像3
# ax3.imshow(img1)
# ax3.axis('off')
#
# # 在第4个子图中显示图像4
# ax4.imshow(img1)
# ax4.axis('off')
#
# ax4.imshow(img1)
# ax4.axis('off')
#
# ax4.imshow(img1)
# ax4.axis('off')

fig = plt.figure(figsize=(12, 6),dpi=600)

# 在1行2列的第1个位置放置子图
ax1 = fig.add_subplot(1, 6, 1)
ax1.imshow(img1)
ax1.axis('off')

# 在1行2列的第2个位置放置子图
ax2 = fig.add_subplot(1,6, 2)
ax2.imshow(img2)
ax2.axis('off')

# 在2行2列的第1个位置放置子图
ax3 = fig.add_subplot(1,6, 3)
ax3.imshow(img3)
ax3.axis('off')

# 在2行2列的第2个位置放置子图
ax4 = fig.add_subplot(1,6, 4)
ax4.imshow(img4)
ax4.axis('off')

ax5 = fig.add_subplot(1,6, 5)
ax5.imshow(img5)
ax5.axis('off')

ax6 = fig.add_subplot(1,6, 6)
ax6.imshow(img6)
ax6.axis('off')

# ax7 = fig.add_subplot(2,4, 7)
# ax7.imshow(img7)
# ax7.axis('off')
#
# ax8 = fig.add_subplot(2,4, 8)
# ax8.imshow(img8)
# ax8.axis('off')

#消除子图之间的间距
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('C:/Users/zxz/Desktop/origin绘图/连接图/merge_DTU(2,3,8,9,10,11).png', bbox_inches='tight')
# 显示画布
plt.show()
# 保存画布

# plt.show()