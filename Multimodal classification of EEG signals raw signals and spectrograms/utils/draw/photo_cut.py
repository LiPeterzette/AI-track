from PIL import Image
import numpy as np
import matplotlib.image as mpimg
# 打开图片并将其转换为 NumPy 数组
import matplotlib.pyplot as plt
def photo_cut(img):
    img_array = np.asarray(img)

    # 获取数组大小和边缘区域的大小
    img_rows, img_cols, img_channels = img_array.shape
    border_size = 100

    # 根据边缘区域大小，计算内部区域的边界
    new_rows = img_rows - 2 * border_size
    new_cols = img_cols - 2 * border_size
    row_start = border_size
    col_start = border_size
    row_end = border_size + new_rows
    col_end = border_size + new_cols

    # 对图片进行裁剪
    img_array = img_array[row_start:row_end, col_start:col_end]

    # 将 NumPy 数组转换为 PIL 图片对象并保存

    return img_array

for i in range(1,17):
    img1 = mpimg.imread(f'C:/Users/zxz/Desktop/origin绘图/DTU/Connection/S{i}_KUL.png')
    img_new = photo_cut(img1)
    plt.imshow(img_new)
    plt.imsave(f'C:/Users/zxz/Desktop/origin绘图/DTU/Connection/S{i}_KUL.png', img_new)
