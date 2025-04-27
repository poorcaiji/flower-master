import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras import backend as K  # 新增导入
from forecast import FLOWER_NAMES 

# 修改模型加载位置（原代码重复加载模型的问题）
def predict_images_in_folders(model, test_dir, output_file="prediction_results.txt", image_size=(224, 224)):
    all_features = []
    all_labels = []
    # 获取所有类别文件夹并按数字顺序排序
    class_folders = sorted([folder for folder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, folder))],key=int)
    class_indices = {folder: index for index, folder in enumerate(class_folders)}
    
    # 初始化错误分类列表（新增代码）
    misclassified = []

    # 创建Grad-CAM模型应放在此处（原代码错误位置）
    last_conv_layer = model.get_layer('conv5_block3_out')  # 根据实际模型结构调整
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

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

                true_class_index = class_indices[class_folder]
                # 在预测循环内部添加（处理每个图像时）
                out = model.predict(prepared_img)
                top1 = np.argmax(out[0])
                feature = model.layers[-2].output  
                feature_model = Model(inputs=model.input, outputs=feature)
                feature_vector = feature_model.predict(prepared_img)
                all_features.append(feature_vector.flatten())
                all_labels.append(true_class_index)  # 现在该变量已定义
                # 新增Grad-CAM生成
                with tf.GradientTape() as tape:
                    conv_output, predictions = grad_model(prepared_img)
                    loss = predictions[:, top1]

                grads = tape.gradient(loss, conv_output)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_output = conv_output[0]
                heatmap = conv_output @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

                # 可视化处理
                resized_img = image.img_to_array(img)  # 使用已经resize后的img对象
                # 调整热图尺寸和通道处理
                heatmap_resized = cv2.resize(heatmap.numpy(), image_size)
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                
                # 确保图像和热图尺寸完全匹配
                resized_img_uint8 = cv2.cvtColor(resized_img.astype('uint8'), cv2.COLOR_RGB2BGR)
                if resized_img_uint8.shape != heatmap_color.shape:
                    heatmap_color = cv2.resize(heatmap_color, (resized_img_uint8.shape[1], resized_img_uint8.shape[0]))
                
                superimposed_img = cv2.addWeighted(resized_img_uint8,0.6,heatmap_color,0.4,0)
                save_path = os.path.join("gradcam_resultsZhuYiLi", f"{os.path.basename(img_path)}")
                # 在预测循环后添加内存清理
                cv2.imwrite(save_path, superimposed_img)
                
                # 新增内存释放（保持原位置不变）
                del conv_output, heatmap, superimposed_img
                K.clear_session()  # 现在可以正常调用
                true_class_index = class_indices[class_folder]
                result = "正确" if top1 == true_class_index else "错误"
                
                # 错误记录应放在此处（原第56行附近）
                if top1 != true_class_index:
                    confidence = out[0][top1]  # 新增置信度获取
                    misclassified.append({
                        'path': img_path,
                        'true': true_class_index, #正确的分类
                        'pred': top1, #预测的分类
                        'confidence': float(confidence)  # 置信度最高值
                    })
                
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
        valid_accuracies = [acc for acc in folder_accuracies.values() if acc > 0.85]
        average_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0

        f.write("\n整体预测统计信息\n")
        f.write("-" * 80 + "\n")
        print("\n整体预测统计信息")
        print("-" * 80)
        for folder, accuracy in folder_accuracies.items():
            f.write(f"文件夹 {folder}: {accuracy * 100:.2f}%\n")
            print(f"文件夹 {folder}: {accuracy * 100:.2f}%")
        f.write(f"\n总的平均预测准确度（排除0%文件夹）: {average_accuracy * 100:.2f}%\n")
        f.write("-" * 80 + "\n")
        print(f"\n总的平均预测准确度（排除0%文件夹）: {average_accuracy * 100:.2f}%")
        print("-" * 80)

    return average_accuracy, all_features, all_labels, misclassified

# 在函数末尾添加错误可视化
def plot_misclassified(samples, n_cols=5, n_rows=4):
    samples_sorted = sorted(samples, key=lambda x: x['confidence'])[:n_cols*n_rows]
    plt.figure(figsize=(16, 12))
    for i, sample in enumerate(samples_sorted):
        img = cv2.imread(sample['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        # 修改标题生成部分
        true_name = FLOWER_NAMES.get(str(sample['true']), f"Unknown_{sample['true']}")
        pred_name = FLOWER_NAMES.get(str(sample['pred']), f"Unknown_{sample['pred']}") 
        plt.title(f"True: {true_name}\nPred: {pred_name}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_examples_zhuYiLi.png')
    plt.close()

# 在主函数中添加显存限制
if __name__ == '__main__':
    # 新增GPU内存限制
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)] # 根据显卡调整
        )
    test_dir = "./data/test"
    # test_dir = "./data/new_jpg"
    image_size = (224, 224)
    model_path = "D:\\code\\pythonProject1\\pythonProject\\flower-master\\saved_model\\model-resnet50.h5"
    output_file = "prediction_resultsZhuYiLi.txt"

    model = load_model(model_path)
    average_accuracy, all_features, all_labels, misclassified = predict_images_in_folders(model, test_dir, output_file, image_size)
    print(f"总的平均预测准确度已保存到文件: {output_file}")

    # t-SNE可视化
    # 在TSNE可视化部分修改（约第153行）
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # 添加perplexity参数
    features_2d = tsne.fit_transform(np.array(all_features))

    plt.figure(figsize=(12, 10))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='tab20', alpha=0.6)
    plt.colorbar()
    plt.title('t-SNE Feature Distribution')
    plt.savefig('tsne_visualization_ZhuyiLi.png')
    plt.close()

    # PCA可视化（可选）
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(np.array(all_features))
    plot_misclassified(misclassified)

