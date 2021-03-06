import os

import torch
import torchvision

from classify import clsfy_img, clsfy_imgs
from model.resnet import Resnet_50

#单个照片判断函数
def single():
    #加载模型和模型参数
    model = torchvision.models.resnet50()
    ckpt = 'models/best_model.ckpt'
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    #请用户输入符合要求的图片
    # print('请输入图片路径')
    file_path = input('请输入图片路径\n')
    if not os.path.isfile(file_path):
        print('该路径不是文件！')
        return
    if not file_path.endswith('.jpg'):
        print('图片必须是jpg格式！')
        return

    #创建六种猫品种字典并分类
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


#多图片判断函数
def multiple():
    # print('请输入文件夹路径')
    folder_path = input('请输入文件夹路径\n')
    if not os.path.isdir(folder_path):
        print('路径必须是文件夹')
        return
    print('分类中')
    contain_njpg_file = clsfy_imgs(folder_path)
    njpg_warn = ''
    #判断是否有不符要求的文件
    if contain_njpg_file:
        njpg_warn = '文件夹中含非jpg文件，这些文件无法被分类'
    print('分类完成，已在文件夹下对图片归类。', njpg_warn)



if __name__ == '__main__':
    #用户操作命令设计
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


