import tkinter
from tkinter import filedialog
import os

import torch
import torchvision

from classify import clsfy_img, clsfy_imgs

#创建猫种类分类tkinter应用程序主窗口
root = tkinter.Tk()
root.title('猫种类分类器')
root['height'] = 140
root['width'] = 200

#在窗口上创建标签组件并放置
clsfy_label = tkinter.Label(root, text='猫种类分类器', width=80)
clsfy_label.grid(row=0, column=0, columnspan=3, pady=20)
path_label = tkinter.Label(root, text='路径:', width='20')
path_label.grid(row=2, column=0)

#分类方式关联的变量，0表示单图片：1表示多图片，默认为单图片
clsfy_type = tkinter.IntVar(value=0)
#设置单图片分类单选按钮
single_ratioB = tkinter.Radiobutton(root, variable=clsfy_type, value=0, text='单图片分类')
single_ratioB.grid(row=1, column=0, pady=20)
#设置多图片分类单选按钮
multiple_ratioB = tkinter.Radiobutton(root, variable=clsfy_type, value=1, text='多图片分类')
multiple_ratioB.grid(row=1, column=1)

#创建路径文本框
path = tkinter.StringVar(root, value='')
path_entry = tkinter.Entry(root, width=50, textvariable=path)
path_entry.grid(row=2, column=1, pady=20)

#浏览函数
def browse():
    if clsfy_type.get() == 0:
        file_path = filedialog.askopenfilename()#获得选择好的文件路径
        path_entry.delete(0, len(path.get()))#删除文本框中原文件路径
        path_entry.insert(0, file_path)#输入新选择好的文件路径
        print(path.get())
    else:
        folder_path = filedialog.askdirectory()#获得选择好的文件夹路径
        path_entry.delete(0, len(path.get()))#删除文本框中原文件夹路径
        path_entry.insert(0, folder_path)#输入新选择好的文件夹路径
        print(path.get())

#分类函数
def clsfy():
    #单图片分类
    if clsfy_type.get() == 0:
        #加载模型及模型参数
        model = torchvision.models.resnet50()
        ckpt = 'models/best_model.ckpt'
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

        #获取文件路径并判断文件是否符合标准
        file_path = path.get()
        if not os.path.isfile(file_path):
            tkinter.messagebox.showinfo(title='提示', message='该路径不是文件！')
            return
        if not file_path.endswith('.jpg'):
            tkinter.messagebox.showinfo(title='提示', message='图片必须是jpg格式！')
            return
        #将文件带入模型计算，输出结果
        type_name_dic = {0: 'americanshorthair',
                         1: 'bengal',
                         2: 'mainecoon',
                         3: 'ragdoll',
                         4: 'scottishfold',
                         5: 'sphinx'
                         }
        type = clsfy_img(model, file_path)
        type_name = type_name_dic[type]
        tkinter.messagebox.showinfo(title='单图片分类结果', message='图片中的猫种类为：{}'.format(type_name))
    #多图片分类
    else:
        #获得文件夹路径并判断是否为文件夹
        folder_path = path.get()
        if not os.path.isdir(folder_path):
            tkinter.messagebox.showinfo(title='提示', message='路径必须是文件夹')
            return
        #将文件夹代入函数计算，检测是否有不符要求的文件，输出结果
        tkinter.messagebox.showinfo(title='开始分类', message='点击确定开始分类，需要一定时间，请稍等')
        contain_njpg_file = clsfy_imgs(folder_path)
        njpg_warn = ''
        if contain_njpg_file:
            njpg_warn = '文件夹中含非jpg文件，这些文件无法被分类'
        tkinter.messagebox.showinfo(title='分类完成', message='分类完成，已在文件夹下对图片归类。 ' + njpg_warn)

#创建浏览与分类按钮并关联相关函数
browse_button = tkinter.Button(root, text='浏览', command=browse)
browse_button.grid(row=2, column=2)
clsfy_button = tkinter.Button(root, text='分类', command=clsfy)
clsfy_button.grid(row=3, column=0, columnspan=3, pady=20)

#启动消息循环
root.mainloop()