import os
import shutil

import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import numpy as np
np.set_printoptions(threshold=np.inf)


def split_train_val_data():
    train_path = 'data/TRAIN'
    val_path = 'data/VAL'
    folder_list = os.listdir(train_path)
    for folder_name in folder_list:
        folder_path = train_path + '/' + folder_name
        file_list = os.listdir(folder_path)
        file_list = sorted(file_list)
        for i in range(150, 200):
            file_name = file_list[i]
            file_path = folder_path + '/' + file_name
            shutil.copyfile(file_path, val_path + '/' + folder_name + '/' + file_name)
            os.remove(file_path)


# 对文件夹内图片重新命名，种类_序号，将格式统一为jpg
def rename_images(folder_path):
    folder_name = folder_path.split('/')[-1]
    file_list = os.listdir(folder_path)
    index = 0
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        new_file_name = folder_name + '_' + str(index) + '.jpg'
        if file_name == new_file_name:
            index += 1
            continue
        new_file_path = folder_path + '/' + new_file_name
        shutil.copyfile(file_path, new_file_path)
        os.remove(file_path)
        index += 1


# 对整个数据集重命名
def rename_data():
    train_folder_list = os.listdir('data/TRAIN')
    val_folder_list = os.listdir('data/VAL')
    test_folder_list = os.listdir('data/TEST')

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        rename_images(train_folder_path)

    for val_folder_name in val_folder_list:
        val_folder_path = 'data/VAL' + '/' + val_folder_name
        rename_images(val_folder_path)

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        rename_images(test_folder_path)


def check_imgs_in_color(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        img = Image.open(file_path)
        if img.mode != 'RGB':
            print(np.array(img).shape, file_path)


def check_data_in_color():
    train_folder_list = os.listdir('data/TRAIN')
    val_folder_list = os.listdir('data/VAL')
    test_folder_list = os.listdir('data/TEST')

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        check_imgs_in_color(train_folder_path)

    for val_folder_name in val_folder_list:
        val_folder_path = 'data/VAL' + '/' + val_folder_name
        check_imgs_in_color(val_folder_path)

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        check_imgs_in_color(test_folder_path)


def img_to_color(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        img = Image.open(file_path)
        if img.mode != 'RGB':
            print(file_path, img.size)
            img = img.convert('RGB')
            print(img.size)
            img.save(fp=file_path)


def data_to_color():
    train_folder_list = os.listdir('data/TRAIN')
    val_folder_list = os.listdir('data/VAL')
    test_folder_list = os.listdir('data/TEST')

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        img_to_color(train_folder_path)

    for val_folder_name in val_folder_list:
        val_folder_path = 'data/VAL' + '/' + val_folder_name
        img_to_color(val_folder_path)

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        img_to_color(test_folder_path)


def get_train_loader(batch_size):
    t_normal = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    t_pad = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Pad(padding=50),
        transforms.ToTensor()
    ])
    t_rotation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor()
    ])
    t_flip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1),
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
    train_set = torch.utils.data.ConcatDataset([normal_set, pad_set, rotation_set, flip_set])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader


def get_vali_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_set = DatasetFolder('data/VAL', loader=lambda x: Image.open(x), extensions="jpg",
                              transform=transform)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return val_loader


def get_test_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_set = DatasetFolder('data/TEST', loader=lambda x: Image.open(x), extensions="jpg",
                              transform=transform)
    # print(test_set.find_classes('data/TEST'))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return test_loader


def test_train_loader():
    train_loader = get_train_loader(1)
    print('val_loader size:', len(train_loader))
    for i, sample in enumerate(train_loader):
        img = sample[0]
        label = sample[1]
        print('img shape:', img.shape)
        print('label shape:', label.shape)
        print(img)
        print(label)
        break


# 测试get_train_val_loader函数
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

