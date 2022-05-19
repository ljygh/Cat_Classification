from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
import random

def readpic(picture):
    if picture.find('americanshorthair')>1:
        return '1_'
    if picture.find('bengal') > 1:
        return '2_'
    if picture.find('mainecoon') > 1:
        return '3_'
    if picture.find('ragdoll') > 1:
        return '4_'
    if picture.find('scottishfold') > 1:
        return '5_'
    if picture.find('sphinx') > 1:
        return '6_'

image = []
paths = r"C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\data"
path1 = 'americanshorthair'
path2 = 'bengal'
path3 = 'mainecoon'
path4 = 'ragdoll'
path5 = 'scottishfold'
path6 = 'sphinx'
notpath = [r'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\data\TEST',r'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\data',r'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\data\TRAIN']

data_pri ={}

for path, subdirs, _ in os.walk(os.path.join(paths)):
    if path in notpath:
        continue
    data_pri[path] = os.listdir(path)
roots = sorted(list(data_pri.keys()))

for i in roots:
    print(i)
    for pathx in data_pri[i]:
        image.append(os.path.join(i,pathx))

random.shuffle(image)
#1500
# test1 = "C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\test1"
# validation1 = "C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\validation1"
# train1 = 'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\train1'
# x = 0
for i in range(0,1500):
    if i < 900:
        numberx = readpic(image[i])
        img_read=cv2.imread(image[i])
        string_name=numberx+str(i+1)+'.jpg'
        cv2.imwrite(os.path.join(r"C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\train1",string_name),img_read)
    if 899 < i < 1200:
        numberx = readpic(image[i])
        img_read = cv2.imread(image[i])
        string_name=numberx+str(i+1)+'.jpg'
        cv2.imwrite(os.path.join(r"C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\validation1",string_name),img_read)
    if i > 1199:
        numberx = readpic(image[i])
        img_read = cv2.imread(image[i])
        string_name=numberx+str(i+1)+'.jpg'
        cv2.imwrite(os.path.join(r'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\test1',string_name),img_read)

#处理数据部分结束

def GaussianNoise(image,percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg



def train_dataloader():
    train_data = []
    x = os.listdir(r'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\train1')
    for i in x:
        image = cv2.imread(r'C:/Users/CHL2454007639/Desktop/Course/python/Cat_Classification/an_data/train1/'+i)
        image_ch = GaussianNoise(image,0.1)
        train_data.append(image_ch)
    train_da = DataLoader(train_data,batch_size=64)
    return train_da

def test_dataloader():
    test_data = []
    x = os.listdir(r'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\test1')
    for i in x:
        image = cv2.imread(r'C:/Users/CHL2454007639/Desktop/Course/python/Cat_Classification/an_data/test1/'+i)
        image_ch = GaussianNoise(image,0.1)
        test_data.append(image_ch)
    test_da = DataLoader(test_data,batch_size=64)
    return test_da

def validation_dataloader():
    va_data = []
    x = os.listdir(r'C:\Users\CHL2454007639\Desktop\Course\python\Cat_Classification\an_data\validation1')
    for i in x:
        image = cv2.imread(r'C:/Users/CHL2454007639/Desktop/Course/python/Cat_Classification/an_data/validation1/'+i)
        image_ch = GaussianNoise(image,0.1)
        va_data.append(image_ch)
    va_da = DataLoader(test_data,batch_size=64)
    return va_da

# 训练集测试集分割
def data_split():
    pass

# 对图片重新命名，种类_序号
def rename_images():
    pass

# 生成train_loader, vali_loader
def get_train_vali_loader():
    pass
    # return train_loader, vali_loader

# 生成test_loader
def get_test_loader():
    pass
    # return test_loader

# 测试函数
def test():
    pass
