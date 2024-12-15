import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 读入四张图片
img1 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/merge_KUL.png')
img2 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/ALL_KUL.png')
img3 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/merge_DTU.png')
img4 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/连接图/ALL_DTU.png')

# 计算每个子图的高度和宽度比例
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]
h4, w4 = img4.shape[:2]

# 计算每行图片的高度比例
h_ratio1 = h1 / h2
h_ratio2 = h3 / h4
max_h_ratio = max(h_ratio1, h_ratio2)
height_ratio = [max_h_ratio, 1]

# 计算第一列和第二列图片的宽度比例
w_ratio1 = w1 / w2
w_ratio2 = w3 / w4

# 创建子图布局
fig, ax = plt.subplots(2, 2, figsize=(8, 8),
                       gridspec_kw={
                           'width_ratios': [w_ratio1, 1],
                           'height_ratios': height_ratio,
                       })


# 调整子图间距和上下留白
plt.subplots_adjust(hspace=0.05, wspace=0.05,
                    bottom=0.1, top=0.95)


# 绘制第一张子图
ax[0, 0].imshow(img1)
ax[0, 0].set_title('Image 1', fontsize=12)
ax[0, 0].axis('off')


# 绘制第二张子图
ax[0, 1].imshow(img2)
ax[0, 1].set_title('Image 2', fontsize=12)
ax[0, 1].axis('off')


# 绘制第三张子图
ax[1, 0].imshow(img3)
ax[1, 0].set_title('Image 3', fontsize=12)
ax[1, 0].axis('off')


# 绘制第四张子图
ax[1, 1].imshow(img4)
ax[1, 1].set_title('Image 4', fontsize=12)
ax[1, 1].axis('off')

# 在每个子图正下方添加标题
ax[0, 0].text(0.5, -0.2, 'This is Image 1',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax[0, 0].transAxes, fontsize=10)

ax[0, 1].text(0.5, -0.2, 'This is Image 2',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax[0, 1].transAxes, fontsize=10)

ax[1, 0].text(0.5, -0.2, 'This is Image 3',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax[1, 0].transAxes, fontsize=10)

ax[1, 1].text(0.5, -0.2, 'This is Image 4',
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax[1, 1].transAxes, fontsize=10)

# 显示绘制结果
plt.show()