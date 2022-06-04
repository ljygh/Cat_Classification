import torch
import torchvision

from classify import clsfy_img, clsfy_imgs
from model.resnet import Resnet_50


def single():
    model = torchvision.models.resnet50()
    ckpt = 'models/036.ckpt'
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # print('请输入图片路径')
    file_path = input('请输入图片路径\n')

    type_name_dic = {0: 'americanshorthair',
                     1: 'bengal',
                     2: 'mainecoon',
                     3: 'ragdoll',
                     4: 'scottishfold',
                     5: 'sphinx'
                     }
    type = clsfy_img(model, file_path)
    type_name = type_name_dic[type]
    print('图片中猫的种类为：', type_name)


def multiple():
    # print('请输入文件夹路径')
    folder_path = input('请输入文件夹路径\n')
    print('分类中')
    clsfy_imgs(folder_path)
    print('分类完成，已在文件夹下对图片归类')



if __name__ == '__main__':
    while (True):
        # print('单张宠物猫图片种类判断请输入1，多张宠物猫图片种类分类请输入2，退出请输入q')
        user_input = input('\n单张宠物猫图片种类判断请输入1，多张宠物猫图片种类分类请输入2，退出请输入q\n')
        if user_input == '1':
            single()
        elif user_input == '2':
            multiple()
        elif user_input == 'q':
            break
        else:
            print('输入有误，请重新输入')


