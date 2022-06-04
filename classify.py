import os
import shutil

import torch
import torchvision.models
from PIL import Image
from torchvision import transforms

from model.resnet import Resnet_50


def clsfy_img(model, file_path):
    model.eval()

    img = Image.open(file_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = torch.unsqueeze(img, 0)

    logits = model(img)
    logit = logits.argmax(dim=-1)
    logit = int(logit)
    return logit


def clsfy_imgs(folder_path):
    model = torchvision.models.resnet50()
    ckpt = 'models/036.ckpt'
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = folder_path + '/' + file_name
        type = clsfy_img(model, file_path)

        type_name_dic = {0 : 'americanshorthair',
                         1 : 'bengal',
                         2 : 'mainecoon',
                         3 : 'ragdoll',
                         4 : 'scottishfold',
                         5 : 'sphinx'
                         }
        type_name = type_name_dic[type]
        if os.path.exists(folder_path + '/' + type_name):
            shutil.copyfile(file_path, folder_path + '/' + type_name + '/' + file_name)
        else:
            os.mkdir(folder_path + '/' + type_name)
            shutil.copyfile(file_path, folder_path + '/' + type_name + '/' + file_name)


def test_clsfy_img():
    model = torchvision.models.resnet50()
    ckpt = 'models/036.ckpt'
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    type = clsfy_img(model, 'data/TEST/americanshorthair/americanshorthair_6.jpg')
    print(type)


def test_clsfy_imgs():
    clsfy_imgs('data/TEST-mix-10')


if __name__ == '__main__':
    # test_clsfy_img()
    test_clsfy_imgs()
