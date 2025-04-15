import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model


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

    return average_accuracy


if __name__ == '__main__':
    test_dir = "./data/test"
    image_size = (224, 224)
    model_path = "./saved_model/model-resnet50.h5"
    output_file = "prediction_results.txt"

    model = load_model(model_path)
    average_accuracy = predict_images_in_folders(model, test_dir, output_file, image_size)
    print(f"总的平均预测准确度已保存到文件: {output_file}")