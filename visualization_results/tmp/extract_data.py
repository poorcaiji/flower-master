import re

# 读取训练日志文件
with open('d:/code/pythonProject1/pythonProject/flower-master/saved_model/training_log.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 提取数据
data = []
for line in lines:
    # 匹配每个Epoch结束时的最终结果
    match = re.search(r'\d+/\d+.*loss: ([\d.]+).*accuracy: ([\d.]+).*val_loss: ([\d.]+).*val_accuracy: ([\d.]+)', line)
    if match:
        loss = float(match.group(1))
        accuracy = float(match.group(2))
        val_loss = float(match.group(3))
        val_accuracy = float(match.group(4))
        data.append([loss, accuracy, val_loss, val_accuracy])

# 保存为CSV文件
with open('d:/code/pythonProject1/pythonProject/flower-master/static/excel/training_log.csv', 'w', encoding='utf-8') as f:
    f.write('批次,损失率,准确率,验证损失率,验证准确率\n')
    for i, row in enumerate(data, start=1):
        f.write(f'{i},{",".join(map(str, row))}\n')