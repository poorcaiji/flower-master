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
    '0': '粉红报春花',
    '1': '硬叶袋兰花',
    '2': '坎特伯雷钟花',
    '3': '香豌豆',
    '4': '英国万寿菊',
    '5': '虎百合',
    '6': '月亮兰花',
    '7': '天堂鸟',
    '8': '乌头花',
    '9': '球状蓟',
    '10': '金鱼草',
    '11': '款冬花',
    '12': '帝王普罗梯亚',
    '13': '长刺蓟',
    '14': '黄鸢尾',
    '15': '球花',
    '16': '紫锥花',
    '17': '秘鲁百合',
    '18': '桔梗',
    '19': '巨型白色天南星',
    '20': '火百合',
    '21': '针垫花',
    '22': '贝母花',
    '23': '红姜花',
    '24': '葡萄风信子',
    '25': '虞美人',
    '26': '威尔士王子羽毛',
    '27': '无茎龙胆',
    '28': '洋蓟',
    '29': '甜威廉',
    '30': '康乃馨',
    '31': '花园福禄考',
    '32': '雾中爱',
    '33': '墨西哥翠菊',
    '34': '高山海冬青',
    '35': '红唇卡特兰',
    '36': '好望角花',
    '37': '大师草',
    '38': '暹罗郁金香',
    '39': '四旬花',
    '40': '巴氏菊花',
    '41': '水仙花',
    '42': '剑兰',
    '43': '一品红',
    '44': '深蓝波列罗',
    '45': '桂竹香',
    '46': '万寿菊',
    '47': '毛茛',
    '48': '牛眼菊',
    '49': '蒲公英',
    '50': '矮牵牛',
    '51': '野生三色堇',
    '52': '报春花',
    '53': '向日葵',
    '54': '天竺葵',
    '55': '兰达夫主教',
    '56': '嘉宝花',
    '57': '天竺葵',
    '58': '橙色大丽花',
    '59': '粉黄大丽花',
    '60': '距花姜',
    '61': '日本银莲花',
    '62': '黑眼苏珊',
    '63': '银叶',
    '64': '加州罂粟',
    '65': '南非菊',
    '66': '春番红花',
    '67': '有须鸢尾',
    '68': '银莲花',
    '69': '树罂粟',
    '70': '勋章菊',
    '71': '杜鹃花',
    '72': '睡莲',
    '73': '玫瑰',
    '74': '曼陀罗',
    '75': '牵牛花',
    '76': '西番莲',
    '77': '莲花',
    '78': '蟾蜍百合',
    '79': '火鹤花',
    '80': '缅栀子花',
    '81': '铁线莲',
    '82': '木槿',
    '83': '耧斗菜',
    '84': '沙漠玫瑰',
    '85': '树锦葵',
    '86': '木兰',
    '87': '仙客来',
    '88': '豆瓣菜',
    '89': '美人蕉',
    '90': '朱顶红',
    '91': '蜜蜂花',
    '92': '球藓',
    '93': '毛地黄',
    '94': '三角梅',
    '95': '山茶花',
    '96': '锦葵',
    '97': '墨西哥矮牵牛',
    '98': '凤梨',
    '99': '天人菊',
    '100': '凌霄花',
    '101': '黑莓百合'
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
    """生成独立的权重分布和样本数量分布折线图"""
    print("正在生成类别权重分析图...")

    # 配置中文字体（全局设置）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据准备
    total_samples = len(labels)
    unique_classes = list(set(labels))
    selected_classes = random.sample(unique_classes, min(20, len(unique_classes)))

    counter = Counter(labels)
    counts = [counter[cls] for cls in selected_classes]
    raw_weights = [total_samples / count for count in counts]
    max_weight = max(raw_weights)
    weights = [(w / max_weight) * 100 for w in raw_weights]
    names = [class_names[cls] for cls in selected_classes]
    # ----------------------------------
    # 图1：权重分布图
    # ----------------------------------
    plt.figure(figsize=(25, 12))
    ax1 = plt.gca()
    # 绘制折线图
    ax1.plot(
        names, weights,
        marker='o', linestyle='-',
        linewidth=3, color='#d62728'
    )
    # 图表装饰
    ax1.set_title("类别权重分布分析", fontsize=24, pad=20)
    ax1.set_xlabel("类别名称", fontsize=24)
    ax1.set_ylabel("归一化权重 (%)", fontsize=24)
    # 横坐标标签设置
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=24,
        weight='bold'  # 加粗字体
    )
    plt.yticks(
        fontsize=24,
        weight='bold'  # 加粗字体
    )
    # 添加数据标签
    max_y = max(weights)
    for x, y in enumerate(weights):
        ax1.text(
            x, y + max_y * 0.01,
            f'{y:.1f}%',
            ha='center', va='bottom',
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "weight_distribution.png"), dpi=120)
    plt.close()
    print("权重分布图已保存")

    # ----------------------------------
    # 图2：样本数量分布图
    # ----------------------------------
    plt.figure(figsize=(25, 12))
    ax2 = plt.gca()
    # 绘制折线图
    ax2.plot(
        names, counts,
        marker='s', linestyle='--',
        linewidth=3, color='#1f77b4'
    )
    # 图表装饰
    ax2.set_title("类别样本数量分布", fontsize=24, pad=20)
    ax2.set_xlabel("类别名称", fontsize=24)
    ax2.set_ylabel("样本数量", fontsize=24)
    # 横坐标标签设置
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=24,
        weight='bold'  # 加粗字体
    )
    plt.yticks(
        fontsize=24,
        weight='bold'  # 加粗字体
    )
    # 添加数据标签
    max_y = max(counts)
    for x, y in enumerate(counts):
        ax2.text(
            x, y + max_y * 0.01,
            f'{y}',
            ha='center', va='bottom',
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_count_distribution.png"), dpi=120)
    plt.close()
    print("样本数量分布图已保存")

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