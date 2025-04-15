# encoding: utf-8
import os
from urllib.request import urlretrieve
import tarfile
from scipy.io import loadmat
from shutil import copyfile
import glob
import numpy as np

# 函数说明：按照分类(labels)拷贝未分组的图片到指定的位置
# Parameters:
#     data_path - 数据存放目录
#     labels - 数据对应的标签，需要按标签放到不同的目录
def copy_data_files(data_path, labels):
    # 检查数据存放目录是否存在，若不存在则创建
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # 创建分类目录，该数据集一共有102个类别
    for i in range(0, 102):
        # 为每个类别创建一个子目录，目录名为类别编号
        os.mkdir(os.path.join(data_path, str(i)))

    # 遍历每个标签信息
    for label in labels:
        # 原图片路径
        src_path = str(label[0])
        # 目的图片路径，根据标签信息将图片放到对应的类别目录下
        dst_path = os.path.join(data_path, str(label[1]), src_path.split(os.sep)[-1])
        # 复制图片文件到目标路径
        copyfile(src_path, dst_path)

if __name__ == '__main__':
    # 检查本地数据集目录是否存在，不存在则创建
    data_set_path = "./data"
    if not os.path.exists(data_set_path):
        os.mkdir(data_set_path)

    # 下载 102 Category Flower 数据集并解压（图片文件）
    flowers_archive_file = "102flowers.tgz"
    # 官网网址
    flowers_url_frefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    flowers_archive_path = os.path.join(data_set_path, flowers_archive_file)
    # 检查图片压缩文件是否存在，若不存在则下载并解压
    if not os.path.exists(flowers_archive_path):
        print("正在下载图片文件...")
        # 从指定网址下载图片压缩文件到本地
        urlretrieve(flowers_url_frefix + flowers_archive_file, flowers_archive_path)
        print("图片文件下载完成.")
        print("正在解压图片文件...")
        # 打开压缩文件并解压到指定目录
        tarfile.open(flowers_archive_path).extractall(path=data_set_path)
        print("图片文件解压完成.")

    # 下载标识文件，标识不同文件的类别（标签文件）
    flowers_labels_file = "imagelabels.mat"
    flowers_labels_path = os.path.join(data_set_path, flowers_labels_file)
    # 检查标签文件是否存在，若不存在则下载
    if not os.path.exists(flowers_labels_path):
        print("正在下载标识文件...")
        # 从指定网址下载标签文件到本地
        urlretrieve(flowers_url_frefix + flowers_labels_file, flowers_labels_path)
        print("标识文件下载完成")
    # 加载标签文件中的标签信息，并将标签值减 1 使其从 0 开始
    flower_labels = loadmat(flowers_labels_path)['labels'][0] - 1

    # 下载数据集分类文件，包含训练集、验证集和测试集（数据集划分文件）
    sets_splits_file = "setid.mat"
    sets_splits_path = os.path.join(data_set_path, sets_splits_file)
    # 检查数据集划分文件是否存在，若不存在则下载
    if not os.path.exists(sets_splits_path):
        print("正在下载数据集分类文件...")
        # 从指定网址下载数据集划分文件到本地
        urlretrieve(flowers_url_frefix + sets_splits_file, sets_splits_path)
        print("数据集分类文件下载完成")
    # 加载数据集划分文件
    sets_splits = loadmat(sets_splits_path)

    # 由于数据集分类文件中测试集数量比训练集多，所以进行了对调，并且把索引值-1，让它们从0开始
    train_set = sets_splits['tstid'][0] - 1
    valid_set = sets_splits['valid'][0] - 1
    test_set = sets_splits['trnid'][0] - 1

    # 获取图片文件名并找到图片对应的分类标识
    image_files = sorted(glob.glob(os.path.join(data_set_path, 'jpg', '*.jpg')))
    # image_labels的大小是[8189, 2]，第一列是图片名称，第二列是花的标签
    image_labels = np.array([i for i in zip(image_files, flower_labels)])

    # 将训练集、验证集和测试集分别放在不同的目录下
    print("正在进行训练集的拷贝...")
    # 调用 copy_data_files 函数将训练集图片拷贝到训练集目录
    copy_data_files(os.path.join(data_set_path, 'train'), image_labels[train_set, :])
    print("已完成训练集的拷贝，开始拷贝验证集...")
    # 调用 copy_data_files 函数将验证集图片拷贝到验证集目录
    copy_data_files(os.path.join(data_set_path, 'valid'), image_labels[valid_set, :])
    print("已完成验证集的拷贝，开始拷贝测试集...")
    # 调用 copy_data_files 函数将测试集图片拷贝到测试集目录
    copy_data_files(os.path.join(data_set_path, 'test'), image_labels[test_set, :])
    print("已完成测试集的拷贝，所有的图片下载和预处理工作已完成.")