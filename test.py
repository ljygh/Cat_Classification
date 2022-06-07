import time

import torch
import torchvision.models
from torch import nn

from data import get_test_loader, get_train_loader, get_vali_loader
from model.resnet import Resnet_50


# 测试模型，生成结果，计算准确率
def test(model, data_loader):
    #设置模型模式，确保BN层中的均值和方差是训练得到的均值方差
    model.eval()
    #记录开始时间
    start_time = time.time()

    #测试数据集并记录结果
    right_pred = 0
    #将数据集分为预测和标签部分
    for i, sample in enumerate(data_loader):#将data_loader组合为一个索引序列，同时列出数据sample和数据下标i
        imgs = sample[0]
        labels = sample[1]
        if torch.cuda.is_available():#GPU是否可用
            imgs = imgs.cuda()
            labels = labels.cuda()

        with torch.no_grad():#无需计算梯度也不会反向传播，避免进行损失梯度的计算
            logits=model(imgs)

        if torch.cuda.is_available():
            logits = logits.argmax(dim=-1).cpu().numpy()#返回相应维度axis上最大值的位置
            labels = labels.cpu().numpy()
        else:
            logits = logits.argmax(dim=-1).numpy()
            labels = labels.numpy()

        #预测与标签比对并记录正确数
        for i in range(0, len(logits)):
            if logits[i] == labels[i]:
                right_pred += 1
    #计算模型准确率
    print(right_pred, len(data_loader.dataset))
    accuracy = right_pred / len(data_loader.dataset)
    #记录结束时间
    end_time = time.time()
    print('Testing Time: %d s, Accuracy: %f' % ((end_time - start_time), accuracy))


if __name__ == '__main__':
    #加载模型和模型参数
    ckpt = 'models/003.ckpt'
    #单次传递给程序的参数个数
    batch_size = 128

    # model = Resnet_50()
    model = torchvision.models.resnet50()
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    print(f"Model loaded from {ckpt}")

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        model = nn.DataParallel(model)

    # train_loader = get_train_loader(batch_size)
    # test(model, train_loader)

    #将验证数据集分割代入模型测试
    val_loader = get_vali_loader(batch_size)
    test(model, val_loader)

    #将测试数据集分割代入模型测试
    test_loader = get_test_loader(batch_size)
    test(model, test_loader)
