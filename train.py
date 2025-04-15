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


# 函数说明：该函数用于重写 DirectoryIterator 的 next 函数，用于将 RGB 通道换成 BGR 通道
def override_keras_directory_iterator_next():
    from tensorflow.keras.preprocessing.image import DirectoryIterator

    original_next = DirectoryIterator.next

    # 防止多次覆盖
    if 'custom_next' in str(original_next):
        return

    def custom_next(self):
        batch_x, batch_y = original_next(self)
        batch_x = batch_x[:, ::-1, :, :]
        return batch_x, batch_y

    DirectoryIterator.next = custom_next


# 函数说明：创建 ResNet50 模型
# Parameters:
#     classes - 所有的类别
#     image_size - 输入图片的尺寸
# Returns:
#     Model - 模型
def create_resnet50_model(classes, image_size):
    # 利用 Keras 的 API 创建模型，并在该模型的基础上进行修改
    # include_top：是否保留顶层的全连接网络;input_tensor: 可选，可填入 Keras tensor 作为模型的输入（即 layers.Input() 输出的 tensor）
    # 权重下载地址：https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # 注意，一定要使用v0.1版本，不能使用v0.2版本，不然会报错
    base_model = ResNet50(include_top=False, input_tensor=Input(shape=image_size + (3,)),
                          weights="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    num = 0
    for layer in base_model.layers:
        num = num + 1
        layer.trainable = False
    print("num = ", num)

    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output = Dense(len(classes), activation='softmax', name='predictions')(x)
    return Model(inputs=base_model.input, outputs=output)


# 函数说明：根据每一类图片的数量不同给每一类图片附上权重
# Parameters:
#     classes - 所有的类别
#     dir - 图片所在的数据集类别的目录，可以是训练集或验证集
# Returns:
#     classes_weight - 每一类的权重
def get_classes_weight(classes, dir):
    class_number = dict()
    # k = 0
    # 获取每一类的图片数量
    for class_name in classes:
        # ！！！原先错误代码，因为classed并不是按照自然顺序排序
        # class_number[k] = len(glob.glob(os.path.join(dir, class_name, '*.jpg')))
        # k += 1
        class_number[int(class_name)] = len(glob.glob(os.path.join(dir, class_name, '*.jpg')))

    # 计算每一类的权重
    total = np.sum(list(class_number.values()))  # 总图片数
    max_samples = np.max(list(class_number.values()))  # 最大的图片数量
    mu = 1. / (total / float(max_samples))
    keys = class_number.keys()
    classes_weight = dict()
    for key in keys:
        # 约等于 ln( float(max_samples)/float(class_number[key]) )
        score = math.log(mu * total / float(class_number[key]))
        classes_weight[key] = score if score > 1. else 1.

    return classes_weight


if __name__ == '__main__':
    # 训练集、验证集、模型输出目录
    train_dir = "./data/train"
    valid_dir = "./data/valid"
    output_dir = "./saved_model"

    # 经过训练后的权重、模型、分类文件
    fine_tuned_weights_path = join_path(output_dir, 'fine-tuned-resnet50-weights.h5')
    weights_path = join_path(output_dir, 'model-weights.h5')
    model_path = join_path(output_dir, 'model-resnet50.h5')
    classes_path = join_path(output_dir, 'classes-resnet50')

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 由于使用 tensorflow 作为 keras 的 backone，所以图片格式设置为 channels_last
    # 修改 DirectoryIterator 的 next 函数，改变 GRB 通道顺序
    # 设置图片数据格式，channels_last表示图片数据的通道在最后一个维度
    K.set_image_data_format('channels_last')
    override_keras_directory_iterator_next()

    # 获取花卉数据类别(不同类别的图片放在不同的目录下，获取目录名即可)
    # classes = sorted([o for o in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, o))])
    classes = sorted([o for o in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, o))], key=int)

    # 获取花卉训练和验证图片的数量
    train_sample_number = len(glob.glob(train_dir + '/**/*.jpg'))
    valid_sample_number = len(glob.glob(valid_dir + '/**/*.jpg'))

    # 创建 Resnet50 模型
    image_size = (224, 224)
    model = create_resnet50_model(classes, image_size)

    # 冻结前 fr_n 层
    fr_n = 10
    for layer in model.layers[:fr_n]:
        layer.trainable = False
    for layer in model.layers[fr_n:]:
        layer.trainable = True

    # 模型配置，使用分类交叉熵作为损失函数，使用 Adam 作为优化器，步长是 1e-5，并使用精确的性能指标
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

    # 获取训练数据和验证数据的 generator
    channels_mean = [103.939, 116.779, 123.68]
    # otation_range: 整数。随机旋转的度数范围。
    # shear_range: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
    # zoom_range: 浮点数 或 [lower, upper]。随机缩放范围。如果是浮点数，[lower, upper] = [1-zoom_range, 1+zoom_range]。
    #  horizontal_flip: 布尔值。随机水平翻转。
    image_data_generator = ImageDataGenerator(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    image_data_generator.mean = np.array(channels_mean, dtype=np.float32).reshape((3, 1, 1))
    # 从文件夹中读取图像
    # 第一个参数是 directory：目标文件夹路径，对于每一个类，该文件夹都要包含一个子文件夹。
    # target_size：整数tuple，默认为(256, 256)。图像将被resize成该尺寸
    # classes：可选参数，为子文件夹的列表，如['cat','dog']，默认为None。若未提供，则该类别列表将从directory下的子文件夹名称/结构自动推断。每一个
    # 子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。
    train_data = image_data_generator.flow_from_directory(train_dir, target_size=image_size, classes=classes)

    image_data_generator = ImageDataGenerator()
    image_data_generator.mean = np.array(channels_mean, dtype=np.float32).reshape((3, 1, 1))
    valid_data = image_data_generator.flow_from_directory(valid_dir, target_size=image_size, classes=classes)

    # 回调函数，用于在训练过程中输出当前进度和设置是否保存过程中的权重，以及早停的判断条件和输出
    # 该回调函数将在每个epoch后保存模型到filepath
    # 第一个参数是保存模型的路径；save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
    # save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    # monitor：需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
    model_checkpoint_callback = ModelCheckpoint(fine_tuned_weights_path, save_best_only=True, save_weights_only=True, monitor='val_loss')
    # early stop是训练模型的过程中，避免过拟合，节省训练时间的一种非常场用的方法。
    # verbose：是否输出更多的调试信息。
    # patience: 在监控指标没有提升的情况下，epochs 等待轮数。等待大于该值监控指标始终没有提升，则提前停止训练。
    # monitor: 监控指标，如val_loss
    early_stopping_callback = EarlyStopping(verbose=1, patience=20, monitor='val_loss')

    # 获取不同类别的权重
    class_weight = get_classes_weight(classes, train_dir)
    batch_size = 10.0
    epoch_number = 50

    print("开始训练...")
    # 利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。
    # 该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
    model.fit(
        train_data,  # 生成器函数
        # steps_per_epoch=train_sample_number / batch_size,  # 每轮的步数，整数，当生成器返回 stesp_per_epoch次数据时，进入下一轮。
        epochs=epoch_number,  # 数据的迭代次数
        validation_data=valid_data,  # 验证集数据
        validation_steps=valid_sample_number / batch_size,  # 当validation_data为生成器时，本参数指定验证集的生成器返回次数
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        class_weight=class_weight  # 规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
        )
    print("模型训练结束，开始保存模型..")
    model.save(model_path)
    model.save_weights(weights_path)
    joblib.dump(classes, classes_path)
    print("模型保存成功，训练任务全部结束.")
    pass