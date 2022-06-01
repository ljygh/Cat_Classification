import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Sequential, ReLU, MaxPool2d, AvgPool2d


class Residual_block(nn.Module):
    def __init__(self, in_channels, first_conv_rate=1, stride=1):
        super(Residual_block, self).__init__()
        assert first_conv_rate in [1, 2, 4]
        if first_conv_rate != 1:
            channels = int(in_channels / first_conv_rate)
        else:
            channels = in_channels

        self.sequence1 = Sequential(
            Conv2d(in_channels, channels, 1),
            BatchNorm2d(channels),
            ReLU(inplace=True)
        )
        self.sequence2 = Sequential(
            Conv2d(channels, channels, 3, stride, 1),
            BatchNorm2d(channels),
            ReLU(inplace=True)
        )
        self.sequence3 = Sequential(
            Conv2d(channels, channels * 4, 1),
            BatchNorm2d(channels * 4),
            ReLU(inplace=True)
        )

        if stride != 1 or in_channels != channels * 4:
            self.change_input_sequence = Sequential(
                Conv2d(in_channels, channels * 4, 3, stride, 1),
                BatchNorm2d(channels * 4),
                ReLU(inplace=True)
            )
        else:
            self.change_input_sequence = None

    def forward(self, x):
        out = self.sequence1(x)
        out = self.sequence2(out)
        out = self.sequence3(out)

        if self.change_input_sequence != None:
            x = self.change_input_sequence(x)

        out = out + x
        return out


class Resnet_50(nn.Module):
    def __init__(self):
        super(Resnet_50, self).__init__()
        self.sequence = Sequential(
            Conv2d(3, 64, 7, 2, 3),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(3, 2)
        )

        layers1 = []
        layers1.append(Residual_block(64))
        for i in range(0, 2):
            layers1.append(Residual_block(256, 4))
        self.block_sequence1 = Sequential(*layers1)

        layers2 = []
        layers2.append(Residual_block(256, 2, 2))
        for i in range(0, 3):
            layers2.append(Residual_block(512, 4))
        self.block_sequence2 = Sequential(*layers2)

        layers3 = []
        layers3.append(Residual_block(512, 2, 2))
        for i in range(0, 5):
            layers3.append(Residual_block(1024, 4))
        self.block_sequence3 = Sequential(*layers3)

        layers4 = []
        layers4.append(Residual_block(1024, 2, 2))
        for i in range(0, 2):
            layers4.append(Residual_block(2048, 4))
        self.block_sequence4 = Sequential(*layers4)

        self.avg_pool = AvgPool2d(7, 1)
        self.fc = nn.Linear(2048, 6)

    def forward(self, x):
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


def test_Residual_block():
    model = Residual_block(512, first_conv_rate=4, stride=1)
    x = torch.autograd.Variable(torch.randn(1, 512, 28, 28))
    out = model(x)
    print(out.shape)


def test_Resnet_50():
    model = Resnet_50()
    x = torch.autograd.Variable(torch.randn(5, 3, 224, 224))
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    # test_Residual_block()
    test_Resnet_50()