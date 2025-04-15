import os
import shutil

# 定义源数据路径和目标路径
BASE_DATA_PATH = r"D:\code\pythonProject1\pythonProject\pythonProject-3\data"
SOURCE_FOLDERS = ["test", "train", "valid"]  # 源数据集文件夹
TARGET_ROOT = os.path.join(BASE_DATA_PATH, "new_jpg")  # 目标根文件夹

def merge_class_images():
    # 创建目标文件夹
    os.makedirs(TARGET_ROOT, exist_ok=True)
    
    # 遍历所有类别（0-101）
    for class_id in range(102):
        class_name = str(class_id)
        target_class_path = os.path.join(TARGET_ROOT, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        # 遍历三个源数据集
        for source in SOURCE_FOLDERS:
            source_class_path = os.path.join(BASE_DATA_PATH, source, class_name)
            if not os.path.exists(source_class_path):
                print(f"警告：源文件夹 {source_class_path} 不存在，跳过")
                continue
            
            # 复制该类别下的所有图片到目标文件夹
            for filename in os.listdir(source_class_path):
                if filename.lower().endswith(('.jpg', '.jpeg')):  # 仅处理JPG格式
                    source_file = os.path.join(source_class_path, filename)
                    target_file = os.path.join(target_class_path, filename)
                    
                    # 复制文件（保留元数据）
                    shutil.copy2(source_file, target_file)
                    print(f"复制文件：{source_file} -> {target_file}")
        
        print(f"类别 {class_name} 图片合并完成，共 {len(os.listdir(target_class_path))} 张图片")

if __name__ == "__main__":
    merge_class_images()
    print("所有类别图片合并完成！")