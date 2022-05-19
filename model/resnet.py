# 复现模型
class Resnet_50():

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dropout, Flatten, Dense

#用resent50学习
model = Sequential()
model.add(ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3), classes=2))

model.summary()






def test():
    from tensorflow.keras.applications.imagenet_utils import decode_predictions

    def pred_img(img):
        #能看猫画图
        plt.imshow(img)
        pit.axis('off')
        plt.show()

        #猫图大小调整
        img_resized = cv2.resize(img,(224,224))
        pred = class Resent_50.predict(img_resized.reshape([1, 224,224, 3]))
        #调整预测结果的对象
        decoed_pred = decode_predictons(pred)

        for i , instance in enumerate(decoded_pred[0]):
            #排名，猫图,预测概率
            print'(第[]: {} ({:.-2f}%))'.format(i+1, instance[1], instance[2] * 100)