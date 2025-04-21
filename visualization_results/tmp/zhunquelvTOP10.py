import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
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


def predict_images_in_folders(model, test_dir, output_file="prediction_results.txt", image_size=(224, 224)):
    # 获取所有类别文件夹并按数字顺序排序
    class_folders = sorted([folder for folder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, folder))], key=int)
    class_indices = {folder: index for index, folder in enumerate(class_folders)}

    total_images = 0
    correct_predictions = 0
    folder_accuracies = {}

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("图片预测结果报告\n")
        f.write("-" * 80 + "\n")
        print("图片预测结果报告")
        print("-" * 80)

        for class_folder in class_folders:
            folder_path = os.path.join(test_dir, class_folder)
            folder_images = os.listdir(folder_path)
            folder_total = len(folder_images)
            folder_correct = 0

            f.write(f"\n正在处理文件夹: {class_folder}\n")
            f.write("-" * 80 + "\n")
            print(f"\n正在处理文件夹: {class_folder}")
            print("-" * 80)

            for img_name in folder_images:
                img_path = os.path.join(folder_path, img_name)
                img = image.load_img(img_path, target_size=image_size)
                img_array = np.expand_dims(image.img_to_array(img), axis=0)
                prepared_img = preprocess_input(img_array)

                out = model.predict(prepared_img)
                top1 = np.argmax(out[0])
                true_class_index = class_indices[class_folder]

                result = "正确" if top1 == true_class_index else "错误"
                f.write(f"图片路径: {img_path}, 预测类别: {top1}, 真实类别: {true_class_index}, 比对结果: {result}\n")
                print(f"图片路径: {img_path}, 预测类别: {top1}, 真实类别: {true_class_index}, 比对结果: {result}")

                total_images += 1
                if top1 == true_class_index:
                    correct_predictions += 1
                    folder_correct += 1

            folder_accuracy = folder_correct / folder_total if folder_total > 0 else 0
            folder_accuracies[class_folder] = folder_accuracy
            f.write(f"文件夹 {class_folder} 的预测准确率: {folder_accuracy * 100:.2f}%\n")
            f.write("-" * 80 + "\n")
            print(f"文件夹 {class_folder} 的预测准确率: {folder_accuracy * 100:.2f}%")
            print("-" * 80)

        # 计算排除80%后的平均准确率
        valid_accuracies = [acc for acc in folder_accuracies.values() if acc > 0.8]
        average_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0

        f.write("\n整体预测统计信息（排除0%文件夹）\n")
        f.write("-" * 80 + "\n")
        print("\n整体预测统计信息（排除0%文件夹）")
        print("-" * 80)
        for folder, accuracy in folder_accuracies.items():
            f.write(f"文件夹 {folder}: {accuracy * 100:.2f}%\n")
            print(f"文件夹 {folder}: {accuracy * 100:.2f}%")
        f.write(f"\n总的平均预测准确度（排除0%文件夹）: {average_accuracy * 100:.2f}%\n")
        f.write("-" * 80 + "\n")
        print(f"\n总的平均预测准确度（排除0%文件夹）: {average_accuracy * 100:.2f}%")
        print("-" * 80)

    # 保存最后 102 种花的名称、英文名称和准确率到 CSV 文件
    last_102_folders = class_folders[-102:]
    csv_output_file = "last_102_flowers_accuracy.csv"
    with open(csv_output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['花卉编号', '花卉英文名称', '准确率']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for folder in last_102_folders:
            flower_english_name = FLOWER_NAMES.get(folder, 'Unknown')
            writer.writerow({'花卉编号': folder, '花卉英文名称': flower_english_name, '准确率': folder_accuracies[folder]})

    print(f"最后 102 种花的编号、英文名称和准确率已保存到 {csv_output_file}")

    return average_accuracy


if __name__ == '__main__':
    test_dir = r"D:\\code\\pythonProject1\\pythonProject\\flower-master\data\\test"
    image_size = (224, 224)
    model_path = r"D:\\code\\pythonProject1\\pythonProject\\flower-master\\other_model\\model-resnet50.h5"
    output_file = "prediction_results.txt"

    model = load_model(model_path)
    average_accuracy = predict_images_in_folders(model, test_dir, output_file, image_size)
    print(f"总的平均预测准确度已保存到文件: {output_file}")