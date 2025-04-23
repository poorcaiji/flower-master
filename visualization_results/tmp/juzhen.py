import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 花卉英文名列表
class_names = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
    'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
    "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower',
    'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
    'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist',
    'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
    'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
    'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
    'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura',
    'geranium', 'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone',
    'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
    'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose',
    'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
    'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen',
    'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea',
    'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
    'blackberry lily'
]

# 读取预测结果文件
file_path = 'D:\\code\\pythonProject1\\pythonProject\\flower-master\\prediction_results.txt'
true_labels = []
pred_labels = []

# 提取预测与真实标签
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(r"预测类别: (\d+), 真实类别: (\d+)", line)
        if match:
            pred = int(match.group(1))
            true = int(match.group(2))
            pred_labels.append(pred)
            true_labels.append(true)

# 构建混淆矩阵
num_classes = 102
cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(num_classes))

# 计算误判次数（t ≠ p）
misclassified_counts = {}
for t, p in zip(true_labels, pred_labels):
    if t != p:
        misclassified_counts[(t, p)] = misclassified_counts.get((t, p), 0) + 1

# 获取误判最多的前10组（真实->预测）
top_10_confusions = sorted(misclassified_counts.items(), key=lambda x: x[1], reverse=True)[:10]

# 提取这些类作为局部混淆矩阵的索引
unique_labels = sorted(set([t for ((t, p), _) in top_10_confusions] + [p for ((t, p), _) in top_10_confusions]))
mini_cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

# 构建小型混淆矩阵
for ((t, p), count) in top_10_confusions:
    i = label_to_idx[t]
    j = label_to_idx[p]
    mini_cm[i, j] = count

# 将热力图数据保存到CSV文件
df_mini_cm = pd.DataFrame(mini_cm, 
                          index=[class_names[i] for i in unique_labels], 
                          columns=[class_names[i] for i in unique_labels])
df_mini_cm.to_csv('D:\\code\\pythonProject1\\pythonProject\\flower-master\\static\\excel\\top_10_confusion_matrix.csv', index=True)

# 可视化热力图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.figure(figsize=(10, 8))
sns.heatmap(mini_cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=[class_names[i] for i in unique_labels], 
            yticklabels=[class_names[i] for i in unique_labels])
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("Top 10易混淆类别的混淆矩阵")
plt.tight_layout()
plt.savefig("D:\\code\\pythonProject1\\pythonProject\\flower-master\\static\\excel\\top10_confusion_matrix.png", dpi=300)
plt.show()