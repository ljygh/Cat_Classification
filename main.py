from model. import model
import cv2
import numpy as np
from PIL import

#选择、读取图片
print('请输入图片名：')
image_name=input()
img=Image.open(image_name)

#读取BGR通道值
image=cv2.imread(image_name)

#读取图片通道数和长宽信息
channel=img.shape[2]
w=img.shape[1]
h=img.shape[0]


#将图片信息带入model比对得到结果
Classification=model.model(image_name)
'''Classification=model.model(channel,w,h)
'''

#输出结果
img.show()
print("该品种为：",Classification)




if __name__ == '__main__':
    main()


