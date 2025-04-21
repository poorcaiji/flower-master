import os
from collections import Counter
import csv

# 导入花卉名称映射表
FLOWER_NAMES = {
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

# 定义图片文件扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

# 定义 new_jpg 文件夹路径
new_jpg_path = r'D:\\code\\pythonProject1\\pythonProject\\flower-master\\data\\new_jpg'

# 检查文件夹是否存在
if not os.path.exists(new_jpg_path):
    print(f"文件夹 {new_jpg_path} 不存在，请检查路径。")
else:
    print(f"文件夹 {new_jpg_path} 存在。")

# 初始化计数器
folder_image_counts = Counter()

# 遍历 new_jpg 下的所有子文件夹
for root, dirs, files in os.walk(new_jpg_path):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        # 统计子文件夹中的图片数量
        image_count = 0
        for file in os.listdir(dir_path):
            if file.lower().endswith(IMAGE_EXTENSIONS):
                image_count += 1
        folder_image_counts[dir_name] = image_count

# 筛选出图片数量在 0 - 300 之间的文件夹
valid_folders = [(folder, count) for folder, count in folder_image_counts.items() if 0 <= count <= 300]

# 如果有效文件夹数量小于 10，直接使用所有有效文件夹
if len(valid_folders) <= 10:
    top_ten_folders = valid_folders
else:
    # 计算均匀分布的间隔
    step = 300 // 9
    target_counts = [i * step for i in range(10)]
    top_ten_folders = []
    for target in target_counts:
        # 找到最接近目标数量的文件夹
        closest_folder = min(valid_folders, key=lambda x: abs(x[1] - target))
        top_ten_folders.append(closest_folder)
        valid_folders.remove(closest_folder)

# 打开 CSV 文件以写入结果
with open('image_count_result.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['文件夹', '花卉类别', '图片数量']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 写入数据
    for folder, count in top_ten_folders:
        flower_name = FLOWER_NAMES.get(folder, '未知类别')
        writer.writerow({'文件夹': folder, '花卉类别': flower_name, '图片数量': count})
        print(f"文件夹: {folder}，花卉类别: {flower_name}，图片数量: {count}")
