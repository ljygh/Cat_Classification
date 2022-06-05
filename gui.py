import tkinter
from tkinter import filedialog
import os

import torch
import torchvision

from classify import clsfy_img, clsfy_imgs

root = tkinter.Tk()
root.title('猫种类分类器')
root['height'] = 140
root['width'] = 200

clsfy_label = tkinter.Label(root, text='猫种类分类器', width=80)
clsfy_label.grid(row=0, column=0, columnspan=3, pady=20)
path_label = tkinter.Label(root, text='路径:', width='20')
path_label.grid(row=2, column=0)

clsfy_type = tkinter.IntVar(value=0)
single_ratioB = tkinter.Radiobutton(root, variable=clsfy_type, value=0, text='单图片分类')
single_ratioB.grid(row=1, column=0, pady=20)
multiple_ratioB = tkinter.Radiobutton(root, variable=clsfy_type, value=1, text='多图片分类')
multiple_ratioB.grid(row=1, column=1)

path = tkinter.StringVar(root, value='')
path_entry = tkinter.Entry(root, width=50, textvariable=path)
path_entry.grid(row=2, column=1, pady=20)

def browse():
    if clsfy_type.get() == 0:
        file_path = filedialog.askopenfilename()
        path_entry.delete(0, len(path.get()))
        path_entry.insert(0, file_path)
        print(path.get())
    else:
        folder_path = filedialog.askdirectory()
        path_entry.delete(0, len(path.get()))
        path_entry.insert(0, folder_path)
        print(path.get())

def clsfy():
    if clsfy_type.get() == 0:
        model = torchvision.models.resnet50()
        ckpt = 'models/best_model.ckpt'
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

        file_path = path.get()
        if not os.path.isfile(file_path):
            tkinter.messagebox.showinfo(title='提示', message='该路径不是文件！')
            return
        if not file_path.endswith('.jpg'):
            tkinter.messagebox.showinfo(title='提示', message='图片必须是jpg格式！')
            return
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
    else:
        folder_path = path.get()
        if not os.path.isdir(folder_path):
            tkinter.messagebox.showinfo(title='提示', message='路径必须是文件夹')
            return
        tkinter.messagebox.showinfo(title='开始分类', message='点击确定开始分类，需要一定时间，请稍等')
        contain_njpg_file = clsfy_imgs(folder_path)
        njpg_warn = ''
        if contain_njpg_file:
            njpg_warn = '文件夹中含非jpg文件，这些文件无法被分类'
        tkinter.messagebox.showinfo(title='分类完成', message='分类完成，已在文件夹下对图片归类。 ' + njpg_warn)

browse_button = tkinter.Button(root, text='浏览', command=browse)
browse_button.grid(row=2, column=2)
clsfy_button = tkinter.Button(root, text='分类', command=clsfy)
clsfy_button.grid(row=3, column=0, columnspan=3, pady=20)

root.mainloop()