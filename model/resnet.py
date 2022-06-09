import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Sequential, ReLU, MaxPool2d, AvgPool2d


class Residual_block(nn.Module): #torch.nn.Module收到python class
    def __init__(self, in_channels, first_conv_rate=1, stride=1):# class 初始化（__init__）
        super(Residual_block, self).__init__()
        assert first_conv_rate in [1, 2, 4] #调试代码
        if first_conv_rate != 1:
            channels = int(in_channels / first_conv_rate)
        else:
            channels = in_channels

        self.sequence1 = Sequential(
            Conv2d(in_channels, channels, 1), #卷积神经网络 (Convolutional Neural Network) 函数
            BatchNorm2d(channels), #批标准化（Batch Normalization）函数
            ReLU(inplace=True) #整流线性单元（Rectified Linear Unit）函数
        )
        self.sequence2 = Sequential(
            Conv2d(channels, channels, 3, stride, 1), #卷积神经网络 (Convolutional Neural Network) 函数
            BatchNorm2d(channels), #批标准化（Batch Normalization）函数
            ReLU(inplace=True) #整流线性单元（Rectified Linear Unit）函数
        )
        self.sequence3 = Sequential(
            Conv2d(channels, channels * 4, 1), #卷积神经网络 (Convolutional Neural Network) 函数
            BatchNorm2d(channels * 4), #批标准化（Batch Normalization）函数
            ReLU(inplace=True) #整流线性单元（Rectified Linear Unit）函数
        )

        if stride != 1 or in_channels != channels * 4:
            self.change_input_sequence = Sequential( #Sequential 模型
                Conv2d(in_channels, channels * 4, 3, stride, 1),
                BatchNorm2d(channels * 4),
                ReLU(inplace=True)
            )
        else:
            self.change_input_sequence = None

    def forward(self, x):#收到数据运算(1)
        out = self.sequence1(x)
        out = self.sequence2(out)
        out = self.sequence3(out)

        if self.change_input_sequence != None:
            x = self.change_input_sequence(x)

        out = out + x
        return out


class Resnet_50(nn.Module):# resnet 50 module
    def __init__(self):
        super(Resnet_50, self).__init__()
        self.sequence = Sequential(
            Conv2d(3, 64, 7, 2, 3),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(3, 2)
        )

        layers1 = [] #layer 1
        layers1.append(Residual_block(64))
        for i in range(0, 2):
            layers1.append(Residual_block(256, 4))
        self.block_sequence1 = Sequential(*layers1)

        layers2 = [] #layer 2
        layers2.append(Residual_block(256, 2, 2))
        for i in range(0, 3):
            layers2.append(Residual_block(512, 4))
        self.block_sequence2 = Sequential(*layers2)

        layers3 = [] #layer 3
        layers3.append(Residual_block(512, 2, 2))
        for i in range(0, 5):
            layers3.append(Residual_block(1024, 4))
        self.block_sequence3 = Sequential(*layers3)

        layers4 = [] #layer 4
        layers4.append(Residual_block(1024, 2, 2))
        for i in range(0, 2):
            layers4.append(Residual_block(2048, 4))
        self.block_sequence4 = Sequential(*layers4)

        self.avg_pool = AvgPool2d(7, 1)
        self.fc = nn.Linear(2048, 6)

    def forward(self, x): #收到数据运算(2)
        x = self.sequence(x)
        x = self.block_sequence1(x)
        x = self.block_sequence2(x)
        x = self.block_sequence3(x)
        x = self.block_sequence4(x)
        x = self.avg_pool(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.fc(x)
        return x


def test_Residual_block():# 测试 学习模块 函数
    model = Residual_block(512, first_conv_rate=4, stride=1)
    x = torch.autograd.Variable(torch.randn(1, 512, 28, 28))
    out = model(x)
    print(out.shape)


def test_Resnet_50():#测试 resnet 50
    model = Resnet_50()
    x = torch.autograd.Variable(torch.randn(5, 3, 224, 224))
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    # test_Residual_block()
    test_Resnet_50()