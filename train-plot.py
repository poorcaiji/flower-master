import numpy as np
import os
import glob
import math
from os.path import join as join_path
import joblib
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (Input, Flatten, Dense, Dropout)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --------------------------- 强制指定 Windows 黑体字体 ---------------------------
# 黑体字体文件路径（Windows 系统内置字体路径）
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
font = FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = font.get_name()  # 设置全局字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# 函数说明：重写 DirectoryIterator 的 next 函数，将 RGB 通道转为 BGR 通道
def override_keras_directory_iterator_next():
    from tensorflow.keras.preprocessing.image import DirectoryIterator
    original_next = DirectoryIterator.next
    if 'custom_next' in str(original_next):
        return

    def custom_next(self):
        batch_x, batch_y = original_next(self)
        batch_x = batch_x[..., ::-1]  # 反转通道顺序（BGR <-> RGB）
        return batch_x, batch_y

    DirectoryIterator.next = custom_next


# 函数说明：创建 ResNet50 模型（加载 ImageNet 预训练权重）
def create_resnet50_model(classes, image_size):
    base_model = ResNet50(
        include_top=False,
        input_tensor=Input(shape=image_size + (3,)),
        weights='imagenet'
    )
    for layer in base_model.layers[:10]:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output = Dense(len(classes), activation='softmax', name='predictions')(x)
    return Model(inputs=base_model.input, outputs=output)


# 函数说明：计算类别权重
def get_classes_weight(classes, dir):
    class_number = {}
    for class_name in classes:
        class_idx = int(class_name)
        class_path = os.path.join(dir, class_name)
        class_number[class_idx] = len(glob.glob(os.path.join(class_path, '*.jpg')))

    total = np.sum(list(class_number.values()))
    max_samples = np.max(list(class_number.values()))
    mu = 1.0 / (total / max_samples)
    classes_weight = {}
    for key in class_number:
        score = math.log(mu * total / class_number[key])
        classes_weight[key] = score if score > 1.0 else 1.0
    return classes_weight


if __name__ == '__main__':
    train_dir = "./data/train"
    valid_dir = "./data/valid"
    output_dir = "./other_model"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    K.set_image_data_format('channels_last')
    override_keras_directory_iterator_next()

    classes = sorted(
        [o for o in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, o))],
        key=lambda x: int(x)
    )

    train_sample_number = len(glob.glob(os.path.join(train_dir, '**/*.jpg'), recursive=True))
    valid_sample_number = len(glob.glob(os.path.join(valid_dir, '**/*.jpg'), recursive=True))
    print(f"训练集样本数: {train_sample_number}, 验证集样本数: {valid_sample_number}")

    image_size = (224, 224)
    model = create_resnet50_model(classes, image_size)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-5),
        metrics=['accuracy']
    )

    channels_mean = [103.939, 116.779, 123.68]

    # 数据生成器
    train_datagen = ImageDataGenerator(
        rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        preprocessing_function=lambda x: x - channels_mean
    )
    train_data = train_datagen.flow_from_directory(
        train_dir, target_size=image_size, classes=classes,
        class_mode='categorical', batch_size=10, shuffle=True
    )

    valid_datagen = ImageDataGenerator(preprocessing_function=lambda x: x - channels_mean)
    valid_data = valid_datagen.flow_from_directory(
        valid_dir, target_size=image_size, classes=classes,
        class_mode='categorical', batch_size=10, shuffle=False
    )

    validation_steps = valid_sample_number // 10
    if validation_steps == 0:
        raise ValueError("验证集样本数不足，至少需要 1 个 batch")

    # 回调函数
    model_checkpoint = ModelCheckpoint(
        join_path(output_dir, 'fine-tuned-resnet50-weights.h5'),
        save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, verbose=1, restore_best_weights=True
    )

    class_weight = get_classes_weight(classes, train_dir)
    epoch_number = 2

    # 模型训练
    history = model.fit(
        train_data,
        epochs=epoch_number,
        validation_data=valid_data,
        validation_steps=validation_steps,
        class_weight=class_weight,
        callbacks=[early_stopping, model_checkpoint]
    )

    # 提取训练结果
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # --------------------------- 简约风格绘图 ---------------------------
    plt.style.use('seaborn-whitegrid')  # 简约白底网格风格
    plt.figure(figsize=(12, 5))  # 缩小图幅

    # 准确率子图
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, marker='o', linestyle='-', linewidth=1.5, label='训练', alpha=0.8, markersize=4)  # 标记点大小调整
    plt.plot(val_acc, marker='s', linestyle='-', linewidth=1.5, label='验证', alpha=0.8, markersize=4)  # 标记点大小调整
    plt.title('准确率曲线', fontproperties=font, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font, fontsize=10)
    plt.ylabel('准确率', fontproperties=font, fontsize=10)
    plt.legend(prop=font, fontsize=9, loc='lower right')
    plt.grid(linestyle='--', alpha=0.6, linewidth=0.8)

    # 损失子图
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, marker='o', linestyle='-', linewidth=1.5, label='训练', alpha=0.8, markersize=4)  # 标记点大小调整
    plt.plot(val_loss, marker='s', linestyle='-', linewidth=1.5, label='验证', alpha=0.8, markersize=4)  # 标记点大小调整
    plt.title('损失曲线', fontproperties=font, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font, fontsize=10)
    plt.ylabel('损失', fontproperties=font, fontsize=10)
    plt.legend(prop=font, fontsize=9, loc='upper right')
    plt.grid(linestyle='--', alpha=0.6, linewidth=0.8)

    plt.suptitle('ResNet50训练结果', fontproperties=font, fontsize=14, y=0.98)  # 简化主标题
    plt.tight_layout(pad=3)  # 调整布局
    plt.savefig(join_path(output_dir, 'train_val_result.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存模型（省略重复代码，保持简洁）
    model.save(join_path(output_dir, 'model-resnet50.h5'))
    joblib.dump(classes, join_path(output_dir, 'classes-resnet50'))
    print("模型训练与保存完成")
