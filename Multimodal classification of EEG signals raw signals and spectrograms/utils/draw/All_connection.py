import matplotlib.pyplot as plt
from matplotlib.image import imread

def create_subplot(images):
    fig, axs = plt.subplots(2, 8, figsize=(10, 3.3))
    for i, img in enumerate(images):
        row = i // 8
        col = i % 8
        axs[row, col].imshow(imread(img))
        axs[row, col].axis('off')
        axs[row, col].set_title(f"S{i+1}",y=-0.25,fontdict={'family' : 'Times New Roman', 'size': 10})
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('C:/Users/zxz/Desktop/origin绘图/DTU/Connection/KUL.png', dpi=750, bbox_inches='tight')
    plt.show()

# 示例用法

image_files = [f'C:/Users/zxz/Desktop/origin绘图/DTU/Connection/S{i}_KUL.png' for i in range(1, 17)]
create_subplot(image_files)
