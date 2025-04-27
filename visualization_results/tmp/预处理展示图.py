import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 图像路径（替换为你的实际路径）
IMAGE_PATH = r"D:\code\pythonProject1\pythonProject\flower-master\data\train\33\image_06930.jpg"
# 保存对比图的文件夹路径
SAVE_DIR = "./visualization_results"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def plot_preprocess_comparison(image_path):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件不存在：{image_path}")

    # ---------------------- 原始图像处理 ----------------------
    img_pil = Image.open(image_path)
    img_pil.load()
    img_rgb_original = image.img_to_array(img_pil)  # 原始RGB数组（任意尺寸）

    # ---------------------- 预处理图像流程 ----------------------
    img_resized = image.load_img(image_path, target_size=(224, 224))
    img_rgb = image.img_to_array(img_resized)
    img_batch = np.expand_dims(img_rgb, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    img_bgr = img_preprocessed[0]

    # 反归一化以便可视化
    mean_values = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_bgr_denorm = img_bgr + mean_values
    img_vis = np.clip(img_bgr_denorm[:, :, ::-1], 0, 255).astype(np.uint8)  # BGR→RGB并截断

    # ---------------------- 创建对比图并保存 ----------------------
    plt.figure(figsize=(12, 6))

    # 左侧：原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.title(f"原始图像\n尺寸：{img_pil.size}")
    plt.axis("off")

    # 右侧：预处理后图像
    plt.subplot(1, 2, 2)
    plt.imshow(img_vis)
    plt.title("预处理后图像\n尺寸：(224, 224)\n(RGB→BGR + 均值减法)")
    plt.axis("off")

    plt.tight_layout()  # 调整子图间距

    # 保存对比图（覆盖原有单独保存的逻辑）
    comparison_save_path = os.path.join(SAVE_DIR, "preprocess_comparison.jpg")
    plt.savefig(comparison_save_path, dpi=300, bbox_inches='tight')  # dpi提高清晰度
    print(f"对比图已保存至：{comparison_save_path}")

    plt.close()  # 关闭图像避免占用内存


if __name__ == "__main__":
    try:
        plot_preprocess_comparison(IMAGE_PATH)
    except Exception as e:
        print(f"出错：{str(e)}")