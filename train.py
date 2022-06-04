import math
import os
import time
import torch
import torchvision.models
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch import nn

from model.resnet import Resnet_50
from data import get_train_loader, get_vali_loader

# 一个epoch训练
def train(**kwargs):
    # 根据参数初始化训练过程中的模型，dataloader，损失函数和优化策略等
    model = kwargs['model']
    train_loader = kwargs['dataloader']
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    epoch = kwargs["epoch"]

    # 设置模型模式
    model.train()

    # 打印epoch信息
    start_time = time.time()
    print('Epoch %03d' % epoch)

    # 训练并记录batch loss
    train_loss = []
    torch.autograd.set_detect_anomaly(True)
    for i, sample in enumerate(train_loader):
        imgs = sample[0]
        labels = sample[1]
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    # 计算并打印训练结果信息
    train_loss = sum(train_loss) / len(train_loss)
    end_time = time.time()
    print('Train batch Loss: %.6f Time: %d s' % (train_loss, end_time - start_time))
    return train_loss


# 一个epoch验证
def validate(**kwargs):
    # 根据参数初始化训练过程中的模型，dataloader，损失函数和epoch序号等
    model = kwargs['model']
    valid_loader = kwargs['dataloader']
    criterion = kwargs['criterion']

    # 设置模型模式
    model.eval()

    # 验证并记录batch loss
    start_time = time.time()
    valid_loss = []
    for i, sample in enumerate(valid_loader):
        imgs = sample[0]
        labels = sample[1]
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            logits = model(imgs)
        loss = criterion(logits, labels)
        valid_loss.append(loss.item())

    # 计算并打印验证结果信息
    valid_loss = sum(valid_loss) / len(valid_loss)
    end_time = time.time()
    print('Validate batch loss: %.6f Time: %d' % (valid_loss, end_time - start_time))
    return valid_loss

# 打印训练过程中train_loss和valid_loss的曲线图
def plot_learning_curve(loss_record, title):
    # 生成横坐标列表
    total_steps = len(loss_record['train'])
    x_1 = x_2 = range(total_steps)

    # 作图
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['validate'], c='tab:cyan', label='val')

    # 设置图例
    plt.ylim(0.0, 5.)
    plt.xlabel('Epoches')
    plt.ylabel('Epoch loss')
    plt.title('Learning curve of {}'.format(title))

    # 显示
    plt.legend()
    plt.show()


# 训练主程序
if __name__ == '__main__':
    # 训练的超参数设置
    mymodel = 'own-50'
    n_epochs = 100
    early_stop = 20
    batch_size = 16
    learning_rate = 0.01
    save_dir = 'models'
    continue_train = None
    save_all = False
    start_epoch = 1

    assert mymodel in ['own-50', 'torch-50-pre', 'torch-50']


    # 初始化模型放入设备
    if mymodel == 'own-50':
        model = Resnet_50()
    elif mymodel == 'torch-50-pre':
        model = torchvision.models.resnet50(pretrained=True)
    elif mymodel == 'torch-50':
        model = torchvision.models.resnet50(pretrained=False)
    else:
        model = None

    if continue_train != None:
        checkpoint = torch.load(continue_train)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        start_epoch = 24
        n_epochs = 100
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        model = nn.DataParallel(model)

    # 初始化损失函数，优化策略和dataloader
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loader = get_train_loader(batch_size)
    valid_loader = get_vali_loader(batch_size)
    train_size = len(train_loader.dataset)
    val_size = len(valid_loader.dataset)

    # 打印训练信息
    print('Start training: Model: {}, Batch size: {}, Learning rate: {}, Early stop: {}, Total epochs: {}, train size: {}, val size: {}'.
        format(mymodel, batch_size, learning_rate, early_stop, n_epochs, train_size, val_size))

    # 训练并记录每个epoch的train_loss和valid_loss
    count = 0
    min_epoch_loss = math.inf
    loss_record = {'train': [],
                   'validate': []}
    for epoch in range(start_epoch, n_epochs + start_epoch):
        # 执行一个epoch的训练与验证并记录loss
        train_loss = train(epoch=epoch,
                           dataloader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer)
        val_loss = validate(dataloader=valid_loader,
                            model=model,
                            criterion=criterion)
        loss_record.get('train').append(train_loss)
        loss_record.get('validate').append(val_loss)

        if val_loss < min_epoch_loss:
            # 如果验证损失小于之前的最小验证损失，保存模型
            min_epoch_loss = val_loss
            count = 0
            print()
            print('Improve! Epoch_loss: {}'.format(val_loss))
            print()
            state_dict = model.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict, },
                os.path.join(save_dir, '%03d.ckpt' % (epoch)))
        else:
            # 否则，更新验证损失没有变小的epoch数，如果大于early_stop结束训练
            count += 1
            if count > early_stop:
                print('Early stop training after {} epoches, {} epoches no update, Min epoch loss: {}'.
                             format(epoch + 1, count, min_epoch_loss))
                break
            if save_all:
                state_dict = model.module.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()
                torch.save({
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict': state_dict, },
                    os.path.join(save_dir, '%03d.ckpt' % (epoch)))


    # 打印训练过程中train_loss和valid_loss的曲线图
    title = 'Model: {}, Learning rate: {}, Batch size: {}'.format('Resnet_50', learning_rate, batch_size)
    plot_learning_curve(loss_record, title)

