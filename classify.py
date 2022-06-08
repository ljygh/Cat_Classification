import os
import shutil

import torch
import torchvision.models
from PIL import Image
from torchvision import transforms

from model.resnet import Resnet_50


def clsfy_img(model, file_path):#将图像设置为适合程序
    model.eval()
    #在程序打开图像
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),# 模型大小转换
        transforms.ToTensor()#数据转换成tensor
    ])
    img = transform(img)#图像转换
    img = torch.unsqueeze(img, 0)

    logits = model(img)
    logit = logits.argmax(dim=-1)#按照1行 → 4行的顺序查找大数字并输出 tensor
    logit = int(logit)
    return logit


def clsfy_imgs(folder_path):
    model = torchvision.models.resnet50() # model = Resnet_50()
    ckpt = 'models/best_model.ckpt'
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    contain_njpg_file = False

    file_list = os.listdir(folder_path)
    for file_name in file_list: #file的类型
        file_path = folder_path + '/' + file_name
        if not file_path.endswith('jpg'):#不是'jpg'结束的情况
            contain_njpg_file = True
            continue
        type = clsfy_img(model, file_path)
        #猫的种类分类
        type_name_dic = {0 : 'americanshorthair',
                         1 : 'bengal',
                         2 : 'mainecoon',
                         3 : 'ragdoll',
                         4 : 'scottishfold',
                         5 : 'sphinx'
                         }
        type_name = type_name_dic[type]
        if os.path.exists(folder_path + '/' + type_name):#确认是否存在 directory 或 file
            shutil.copyfile(file_path, folder_path + '/' + type_name + '/' + file_name)
        else:
            os.mkdir(folder_path + '/' + type_name)#生成folder
            shutil.copyfile(file_path, folder_path + '/' + type_name + '/' + file_name)
    return contain_njpg_file


def test_clsfy_img():
    model = torchvision.models.resnet50() # model = Resnet_50()
    ckpt = 'models/best_model.ckpt'
    checkpoint = torch.load(ckpt)#生成 checkpoint
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict) #导入已经保存的信息
    type = clsfy_img(model, 'data/TEST/americanshorthair/americanshorthair_6.jpg')
    print(type)


def test_clsfy_imgs():
    clsfy_imgs('data/TEST-mix-10')


if __name__ == '__main__':
    # test_clsfy_img()
    test_clsfy_imgs() #看猫图像
