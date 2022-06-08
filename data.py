import os
import shutil

import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import numpy as np
np.set_printoptions(threshold=np.inf)


# 原数据kaggle上下载，分为TRAIN，TEST两个文件夹，此函数将TRAIN文件中的一部分图片抽取出来放到验证集
def split_train_val_data():
    train_path = 'data/TRAIN'
    val_path = 'data/VAL'
    folder_list = os.listdir(train_path)# 将训练集下的文件名（猫的种类文件名）返回成一个列表
    for folder_name in folder_list:
        folder_path = train_path + '/' + folder_name# 定义文件夹路径为训练集/训练集下的每个文件名
        file_list = os.listdir(folder_path)# 将图片名返回成一个列表
        file_list = sorted(file_list)# 进行排序
        for i in range(150, 200):
            file_name = file_list[i]# 定义图片名
            file_path = folder_path + '/' + file_name
            shutil.copyfile(file_path, val_path + '/' + folder_name + '/' + file_name)# 复制图片到验证集中
            os.remove(file_path)


# 对文件夹内图片重新命名，种类_序号，将格式统一为jpg
def rename_images(folder_path):
    folder_name = folder_path.split('/')[-1]# 取列表中倒数第一个元素
    file_list = os.listdir(folder_path)
    index = 0
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        new_file_name = folder_name + '_' + str(index) + '.jpg'# 图片名称为猫的种类_[序号].jpg
        if file_name == new_file_name:
            index += 1
            continue
        new_file_path = folder_path + '/' + new_file_name
        shutil.copyfile(file_path, new_file_path)
        os.remove(file_path)
        index += 1


# 对整个数据集重命名
def rename_data():
    train_folder_list = os.listdir('data/TRAIN')# 返回训练集下文件名称列表
    val_folder_list = os.listdir('data/VAL')# 返回验证集下文件名称列表
    test_folder_list = os.listdir('data/TEST')# 返回测试集下文件名称列表

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        rename_images(train_folder_path)# 调用rename_images()函数将训练集图片重命名

    for val_folder_name in val_folder_list:
        val_folder_path = 'data/VAL' + '/' + val_folder_name
        rename_images(val_folder_path)# 调用rename_images()函数将验证集图片重命名

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        rename_images(test_folder_path)# 调用rename_images()函数将测试集图片重命名

# 检查图片是否为jpg格式
def check_imgs_in_color(folder_path):
    file_list = os.listdir(folder_path)# 将文件下的图片名返回成列表
    for file_name in file_list:
        file_path = folder_path + '/' + file_name# 完整的图片路径
        img = Image.open(file_path)# 打开图片
        if img.mode != 'RGB':# 判断是否为jpg格式，如果不是输出图片大小和路径
            print(np.array(img).shape, file_path)


def check_data_in_color():
    train_folder_list = os.listdir('data/TRAIN')# 返回训练集下文件名称列表
    val_folder_list = os.listdir('data/VAL')# 返回验证集下文件名称列表
    test_folder_list = os.listdir('data/TEST')# 返回测试集下文件名称列表

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        check_imgs_in_color(train_folder_path)# 调用check_imgs_in_color()函数判断训练集图片是否为jpg格式

    for val_folder_name in val_folder_list:
        val_folder_path = 'data/VAL' + '/' + val_folder_name
        check_imgs_in_color(val_folder_path)# 调用check_imgs_in_color()函数判断验证集图片是否为jpg格式

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        check_imgs_in_color(test_folder_path)# 调用check_imgs_in_color()函数判断测试集图片是否为jpg格式

# 将非RGB格式的图片转为RGB格式
def img_to_color(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = folder_path + '/' + file_name# 完整的图片路径
        img = Image.open(file_path)
        if img.mode != 'RGB':# 如果不是RGB格式，就输出图片路径和大小，并转为RGB格式，最后保存
            print(file_path, img.size)
            img = img.convert('RGB')
            print(img.size)
            img.save(fp=file_path)


def data_to_color():
    train_folder_list = os.listdir('data/TRAIN')# 返回训练集下文件名称列表
    val_folder_list = os.listdir('data/VAL')# 返回验证集下文件名称列表
    test_folder_list = os.listdir('data/TEST')# 返回测试集下文件名称列表

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        img_to_color(train_folder_path)# 调用img_to_color()函数将训练集下非RGB格式的图片转为RGB格式

    for val_folder_name in val_folder_list:
        val_folder_path = 'data/VAL' + '/' + val_folder_name
        img_to_color(val_folder_path)# 调用img_to_color()函数将验证集下非RGB格式的图片转为RGB格式

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        img_to_color(test_folder_path)# 调用img_to_color()函数将测试集下非RGB格式的图片转为RGB格式

# 将图片做各种变化
def get_train_loader(batch_size):
    t_normal = transforms.Compose([
        transforms.Resize((224, 224)), # 把给定的图片resize到given size
        transforms.ToTensor()
    ])
    t_pad = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Pad(padding=30),# 先填充30再变为224X224
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    t_rotation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(0, 180)), # 随机转换角度
        transforms.ToTensor()
    ])
    t_flip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1),# 图像随机水平翻转
        transforms.ToTensor()
    ])

    normal_set = DatasetFolder('data/TRAIN', loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=t_normal)
    pad_set = DatasetFolder('data/TRAIN', loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=t_pad)
    rotation_set = DatasetFolder('data/TRAIN', loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=t_rotation)
    flip_set = DatasetFolder('data/TRAIN', loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=t_flip)

    train_set = torch.utils.data.ConcatDataset([normal_set, pad_set, rotation_set, flip_set])# 将图片做对应处理后的集合合在一起
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)# 训练集图片load，一次从数据集中取batch_size个图片放进网络中处理
    return train_loader


def get_vali_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),# 对验证集图片做处理
        transforms.ToTensor()
    ])
    val_set = DatasetFolder('data/VAL', loader=lambda x: Image.open(x), extensions="jpg",
                              transform=transform)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)# 验证集图片load，一次从数据集中取batch_size个图片放进网络中处理
    return val_loader


def get_test_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_set = DatasetFolder('data/TEST', loader=lambda x: Image.open(x), extensions="jpg",
                              transform=transform)
    # print(test_set.find_classes('data/TEST'))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)# 测试集图片load，一次从数据集中取batch_size个图片放进网络中处理
    return test_loader

# 检测get_train_loader()函数能不能读取图片，能不能提取标签
def test_train_loader():
    train_loader = get_train_loader(1)
    print('val_loader size:', len(train_loader))# 输出有多少条数据
    for i, sample in enumerate(train_loader):
        img = sample[0]
        label = sample[1]
        print('img shape:', img.shape)# 输出img，是一个tensor类型
        print('label shape:', label.shape)# 输出label值，表示类别
        print(img)
        print(label)
        break


# 测试get_val_loader函数
def test_val_loader():
    val_loader = get_vali_loader(1)
    print('val_loader size:', len(val_loader))
    for i, sample in enumerate(val_loader):
        img = sample[0]
        label = sample[1]
        print('img shape:', img.shape)
        print('label shape:', label.shape)
        print(img)
        print(label)
        break


# 测试get_test_loader函数
def test_test_loader():
    test_loader = get_test_loader(1)
    print('test_loader size:', len(test_loader))
    for i, sample in enumerate(test_loader):
        img = sample[0]
        label = sample[1]
        print('img shape:', img.shape)
        print('label shape:', label.shape)
        print(img)
        print(label)
        break


if __name__ == '__main__':
    # split_train_val_data()
    # rename_images('data/TRAIN/americanshorthair')
    # rename_data()
    # check_data_in_color()
    # img_to_color('data/TRAIN/sphinx')
    # data_to_color()
    test_train_loader()
    # test_val_loader()
    # test_test_loader()

