import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def visualize_augmentation(image_path, augmentor, save_path=None):
    # 加载原图（PIL 类型）
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 生成增强图像，并记录相应的增强方法
    augmented_images = []
    augmentations = ["旋转", "剪切", "缩放", "翻转"]

    # 使用 ImageDataGenerator 的 flow 方法来生成增强图像
    # 我们需要将图像数组转换成批次，所以使用 np.expand_dims 来处理图像数据
    img_array = np.expand_dims(img_array[0], axis=0)  # 将图像数组变成 (1, 224, 224, 3) 格式

    # 使用 `flow()` 来增强图像
    iterator = augmentor.flow(img_array, batch_size=1)

    # 获取4个增强后的图像
    for i in range(4):
        augmented_img = next(iterator)[0].astype("uint8")
        augmented_images.append(augmented_img)

    # 显示原图 + 增强图像
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(img)  # ✅ 直接使用 PIL 图像
    plt.title("原图")
    plt.axis("off")

    for i, aug_img in enumerate(augmented_images):
        plt.subplot(1, 5, i + 2)
        plt.imshow(aug_img)
        plt.title(f"增强{i+1} - {augmentations[i]}")
        plt.axis("off")

    plt.tight_layout()

    # 创建保存目录（如果不存在）
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300)

    plt.show()

# 示例用法
augmentor = ImageDataGenerator(
    rotation_range=30.,  # 旋转范围
    shear_range=0.2,     # 剪切范围
    zoom_range=0.2,      # 缩放范围
    horizontal_flip=True # 水平翻转
)

# 示例图片路径（换成你某一类的某张图片）
image_path = './data/train/17/image_04249.jpg'
visualize_augmentation(image_path, augmentor, save_path='./visualization_results/sample_augment.png')