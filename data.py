import os
import shutil
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import numpy as np

# 对文件夹内图片重新命名，种类_序号，将格式统一为jpg
def rename_images(folder_path):
    folder_name = folder_path.split('/')[-1]
    file_list = os.listdir(folder_path)
    index = 0
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        new_file_name = folder_name + '_' + str(index) + '.jpg'
        new_file_path = folder_path + '/' + new_file_name
        shutil.copyfile(file_path, new_file_path)
        os.remove(file_path)
        index += 1


# 对整个数据集重命名
def rename_data():
    train_folder_list = os.listdir('data/TRAIN')
    test_folder_list = os.listdir('data/TEST')

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        rename_images(train_folder_path)

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        rename_images(test_folder_path)


def img_to_color(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        img = Image.open(file_path)
        if img.mode != 'RGB':
            print(file_name, img.size)
            img = img.convert('RGB')
            print(img.size)
            img.save(fp=file_path)


def data_to_color():
    train_folder_list = os.listdir('data/TRAIN')
    test_folder_list = os.listdir('data/TEST')

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        img_to_color(train_folder_path)

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        img_to_color(test_folder_path)


def check_imgs_in_color(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        img = Image.open(file_path)
        if img.mode != 'RGB':
            print(np.array(img).shape, file_name)


def check_data_in_color():
    train_folder_list = os.listdir('data/TRAIN')
    test_folder_list = os.listdir('data/TEST')

    for train_folder_name in train_folder_list:
        train_folder_path = 'data/TRAIN' + '/' + train_folder_name
        check_imgs_in_color(train_folder_path)

    for test_folder_name in test_folder_list:
        test_folder_path = 'data/TEST' + '/' + test_folder_name
        check_imgs_in_color(test_folder_path)


# 生成train_loader, vali_loader
def get_train_vali_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_val_set = DatasetFolder('data/TRAIN', loader=lambda x: Image.open(x), extensions="jpg",
                              transform=transform)
    train_size = int(len(train_val_set) * 0.75)
    val_size = len(train_val_set) - train_size
    train_set, val_set = random_split(train_val_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# 生成test_loader
def get_test_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_set = DatasetFolder('data/TEST', loader=lambda x: Image.open(x), extensions="jpg",
                              transform=transform)
    print(test_set.find_classes('data/TEST'))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return test_loader


# 测试get_train_val_loader函数
def test_train_val_loader():
    train_loader, val_loader = get_train_vali_loader(5)
    print('train_loader size:', len(train_loader), 'val_loader size:', len(val_loader))
    for i, sample in enumerate(train_loader):
        img = sample[0]
        label = sample[1]
        print('img shape:', img.shape)
        print('label shape:', label.shape)
        # print(img)
        # print(label)
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
    # rename_images('data/TRAIN/americanshorthair')
    # rename_data()
    # test_train_val_loader()
    # test_test_loader()
    # check_data_in_color()
    # img_to_color('data/TRAIN/sphinx')
    # data_to_color()
    get_test_loader(1)

