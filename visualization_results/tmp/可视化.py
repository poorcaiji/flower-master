from collections import Counter
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 中文用黑体，英文用Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 花卉中文名称映射表
FLOWER_NAMES_CN = {
    '0': 'pink primrose',
    '1': 'hard-leaved pocket orchid',
    '2': 'canterbury bells',
    '3': 'sweet pea',
    '4': 'english marigold',
    '5': 'tiger lily',
    '6': 'moon orchid',
    '7': 'bird of paradise',
    '8': 'monkshood',
    '9': 'globe thistle',
    '10': 'snapdragon',
    '11': "colt's foot",
    '12': 'king protea',
    '13': 'spear thistle',
    '14': 'yellow iris',
    '15': 'globe-flower',
    '16': 'purple coneflower',
    '17': 'peruvian lily',
    '18': 'balloon flower',
    '19': 'giant white arum lily',
    '20': 'fire lily',
    '21': 'pincushion flower',
    '22': 'fritillary',
    '23': 'red ginger',
    '24': 'grape hyacinth',
    '25': 'corn poppy',
    '26': 'prince of wales feathers',
    '27': 'stemless gentian',
    '28': 'artichoke',
    '29': 'sweet william',
    '30': 'carnation',
    '31': 'garden phlox',
    '32': 'love in the mist',
    '33': 'mexican aster',
    '34': 'alpine sea holly',
    '35': 'ruby-lipped cattleya',
    '36': 'cape flower',
    '37': 'great masterwort',
    '38': 'siam tulip',
    '39': 'lenten rose',
    '40': 'barbeton daisy',
    '41': 'daffodil',
    '42': 'sword lily',
    '43': 'poinsettia',
    '44': 'bolero deep blue',
    '45': 'wallflower',
    '46': 'marigold',
    '47': 'buttercup',
    '48': 'oxeye daisy',
    '49': 'common dandelion',
    '50': 'petunia',
    '51': 'wild pansy',
    '52': 'primula',
    '53': 'sunflower',
    '54': 'pelargonium',
    '55': 'bishop of llandaff',
    '56': 'gaura',
    '57': 'geranium',
    '58': 'orange dahlia',
    '59': 'pink-yellow dahlia',
    '60': 'cautleya spicata',
    '61': 'japanese anemone',
    '62': 'black-eyed susan',
    '63': 'silverbush',
    '64': 'californian poppy',
    '65': 'osteospermum',
    '66': 'spring crocus',
    '67': 'bearded iris',
    '68': 'windflower',
    '69': 'tree poppy',
    '70': 'gazania',
    '71': 'azalea',
    '72': 'water lily',
    '73': 'rose',
    '74': 'thorn apple',
    '75': 'morning glory',
    '76': 'passion flower',
    '77': 'lotus',
    '78': 'toad lily',
    '79': 'anthurium',
    '80': 'frangipani',
    '81': 'clematis',
    '82': 'hibiscus',
    '83': 'columbine',
    '84': 'desert-rose',
    '85': 'tree mallow',
    '86': 'magnolia',
    '87': 'cyclamen',
    '88': 'watercress',
    '89': 'canna lily',
    '90': 'hippeastrum',
    '91': 'bee balm',
    '92': 'ball moss',
    '93': 'foxglove',
    '94': 'bougainvillea',
    '95': 'camellia',
    '96': 'mallow',
    '97': 'mexican petunia',
    '98': 'bromelia',
    '99': 'blanket flower',
    '100': 'trumpet creeper',
    '101': 'blackberry lily'
}

# 定义数据路径
TRAIN_DIR = r"D:\code\pythonProject1\pythonProject\flower-master\data\new_jpg"

# 创建保存结果图片的目录
RESULTS_DIR = ".."
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_data():
    """加载数据集并返回图片路径和标签（映射中文名称）"""
    print("正在加载数据...")
    image_paths = []
    labels = []
    class_folders = []  # 存储文件夹名称（0-101字符串）
    
    # 获取所有类别文件夹名称（0-101）
    for folder in sorted(os.listdir(TRAIN_DIR)):
        folder_path = os.path.join(TRAIN_DIR, folder)
        if os.path.isdir(folder_path):
            class_folders.append(folder)  # 文件夹名称为'0','1',...,'101'
    
    # 生成中文类别名称
    chinese_class_names = [FLOWER_NAMES_CN[folder] for folder in class_folders]
    
    # 读取所有图片
    for class_idx, folder in enumerate(class_folders):
        class_path = os.path.join(TRAIN_DIR, folder)
        print(f"正在处理类别 {chinese_class_names[class_idx]}...")
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    print(f"加载完成! 共 {len(image_paths)} 张图片, {len(chinese_class_names)} 个类别")
    return image_paths, labels, chinese_class_names  # 返回中文名称列表

def plot_class_distribution(labels, class_names):
    """绘制各类花的样本分布图（显示中文名称）"""
    print("正在生成类别分布图...")
    label_counts = Counter(labels)
    class_indices = range(len(class_names))
    counts = [label_counts[idx] for idx in class_indices]
    
    plt.figure(figsize=(24, 10))  # 增大图幅适应中文标签
    bars = plt.bar(class_indices, counts, color='#1f77b4', edgecolor='white')
    
    # 显示全部中文名称，自动调整旋转角度避免重叠
    plt.xticks(class_indices, class_names, rotation=45, ha='right', fontsize=8)  # 倾斜45度
    
    plt.xlabel("花的类别", fontsize=14)
    plt.ylabel("样本数量", fontsize=14)
    plt.title("Oxford 102 Flowers 各类样本数量分布", fontsize=16, pad=20)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "类别分布图.png"))
    plt.close()
    print("类别分布图已保存")

def show_samples_per_class(image_paths, labels, class_names, samples_per_class=1):
    """每类花的样例图像（显示中文名称）"""
    print("正在生成每类花的样例图像...")
    class_to_paths = {}
    for path, label in zip(image_paths, labels):
        if label not in class_to_paths:
            class_to_paths[label] = []
        class_to_paths[label].append(path)

    n_classes = len(class_names)
    n_cols = 3  # 每行显示 3 个类别
    n_rows = 2  # 显示 2 行

    plt.figure(figsize=(15, 10))  # 调整图片大小

    selected_classes = list(range(min(6, n_classes)))  
    for i, class_idx in enumerate(selected_classes):
        if class_idx not in class_to_paths or not class_to_paths[class_idx]:
            continue
        selected_path = class_to_paths[class_idx][4]

        plt.subplot(n_rows, n_cols, i + 1)
        try:
            img = Image.open(selected_path)
            plt.imshow(img)
            plt.title(f"{class_names[class_idx]}", fontsize=20, pad=15)  # 增大字体大小
            plt.axis('off')
        except Exception as e:
            print(f"读取图片 {selected_path} 失败: {e}")

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(RESULTS_DIR, "每类花的样例图像.png"))
    plt.close()
    print("每类花的样例图像已保存")


import random
from PIL import Image
import matplotlib.pyplot as plt
import os

# 假设 RESULTS_DIR 已经定义
RESULTS_DIR = '..'

def show_samples_per_class_224(image_paths, labels, class_names, samples_per_class=1):
    """每类花的样例图像（显示中文名称），图片尺寸为224x224"""
    print("正在生成每类花的样例图像...")
    class_to_paths = {}
    for path, label in zip(image_paths, labels):
        if label not in class_to_paths:
            class_to_paths[label] = []
        class_to_paths[label].append(path)

    n_classes = len(class_names)
    n_cols = 3  # 每行显示 3 个类别
    n_rows = 2  # 显示 2 行

    plt.figure(figsize=(15, 10))  # 调整图片大小

    # 直接选取类别索引为 0 到 5 的图片
    selected_classes = list(range(min(6, n_classes)))
    for i, class_idx in enumerate(selected_classes):
        if class_idx not in class_to_paths or not class_to_paths[class_idx]:
            continue
        selected_path = class_to_paths[class_idx][4]
        plt.subplot(n_rows, n_cols, i + 1)
        try:
            img = Image.open(selected_path)
            plt.imshow(img)
            plt.title(f"{class_names[class_idx]}", fontsize=20, pad=15)  # 增大字体大小
            plt.axis('off')
        except Exception as e:
            print(f"读取图片 {selected_path} 失败: {e}")

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(RESULTS_DIR, "每类花的样例图像_224.png"))
    plt.close()
    print("每类花的样例图像已保存")

def plot_random_samples(image_paths, class_names_dict):
    """随机选取6张图片并保存为2x3布局"""
    print("正在生成随机采样图像...")
    # 随机选择6张不同图片
    selected_paths = random.sample(image_paths, 6)
    
    plt.figure(figsize=(15, 10))
    
    # 创建2行3列布局
    for i, img_path in enumerate(selected_paths):
        plt.subplot(2, 3, i+1)
        try:
            # 从路径中提取文件夹名称（类别ID）
            folder_name = os.path.basename(os.path.dirname(img_path))
            # 获取对应的花卉英文名称
            flower_name = class_names_dict.get(folder_name, f"Unknown_{folder_name}")
            
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"{flower_name}", fontsize=20, pad=15)
            plt.axis('off')
        except Exception as e:
            print(f"读取图片 {img_path} 失败: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "random_samples.png"))
    plt.close()
    print("随机采样图像已保存")

def plot_class_weights(labels, class_names):
    """生成双折线图分析（修复中文显示、数据标签和坐标倾斜）"""
    print("正在生成双折线图分析...")
    # 正确设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 中文用黑体，英文用Times New Roman
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示

    total_samples = len(labels)
    unique_classes = list(set(labels))
    selected_classes = random.sample(unique_classes, min(20, len(unique_classes)))

    counter = Counter(labels)
    counts = [counter[cls] for cls in selected_classes]
    # 计算权重
    raw_weights = [(total_samples / count) for count in counts]
    # 归一化权重到 0 - 100% 的范围
    max_weight = max(raw_weights)
    weights = [(weight / max_weight) * 100 for weight in raw_weights]
    names = [class_names[cls] for cls in selected_classes]  # 获取中文类别名称

    plt.figure(figsize=(30, 12))

    # 子图1：逆向权重分布
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(names, weights, marker='o', linestyle='-', linewidth=3, color='#d62728')
    ax1.set_title("权重分布分析", fontsize=24, pad=20)
    ax1.set_xlabel("类别名称", fontsize=20)
    ax1.set_ylabel("权重", fontsize=20)

    # 子图2：样本数量分布
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(names, counts, marker='s', linestyle='--', linewidth=3, color='#1f77b4')
    ax2.set_title("类别样本数量分布", fontsize=24, pad=20)
    ax2.set_xlabel("类别名称", fontsize=20)
    ax2.set_ylabel("样本数量", fontsize=20)

     # 统一设置横坐标倾斜和右对齐，并加粗坐标值
    for ax in [ax1, ax2]:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")
            tick.set_weight('bold')  # 设置字体加粗
            tick.set_fontsize(16)  # 设置字体大小

    # 给权重子图添加标签，以百分比形式显示权重
    for x, y, count in zip(range(len(names)), weights, counts):
        ax1.text(x, y + max(weights) * 0.03,  # 调整y偏移量避免超出边界
                 f'权重: {y:.1f}%',  #权重以百分比形式显示
                 ha='center', va='bottom', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))  # 白色背景框

    # 给样本数量子图添加标签
    for x, y in enumerate(counts):
        ax2.text(x, y + max(counts) * 0.03,  # 调整y偏移量
                 f'{y}',
                 ha='center', va='bottom', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    plt.tight_layout(pad=5)  # 增加子图间距
    plt.savefig(os.path.join(RESULTS_DIR, "inverse_weight_analysis.png"), dpi=120)
    plt.close()
    print("逆向权重分析图已保存")

if __name__ == "__main__":
    # 加载数据（直接返回中文类别名称）
    image_paths, labels, class_names = load_data()
    
    # 生成可视化图表（使用中文名称）
    # plot_class_distribution(labels, class_names)
    # plot_image_size_distribution(image_paths)
    # show_samples_per_class(image_paths, labels, class_names)
    # show_samples_per_class_224(image_paths, labels, class_names)
    # plot_random_samples(image_paths, FLOWER_NAMES_CN)
    # plot_train_test_distribution(len(image_paths))
    # plot_training_curves()
    # plot_confusion_matrix(class_names, n_classes=min(20, len(class_names)))
    plot_class_weights(labels, class_names)
    print("所有可视化图表已生成完毕，保存在 visualization_results 目录下。")