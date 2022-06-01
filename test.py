import time

import torch
from torch import nn

from data import get_test_loader
from model.resnet import Resnet_50


# 测试模型，生成结果，计算准确率
def test(model, test_loader):
    model.eval()
    start_time = time.time()

    right_pred = 0
    for i, sample in enumerate(test_loader):
        imgs = sample[0]
        labels = sample[1]
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            logits=model(imgs)

        if torch.cuda.is_available():
            logits = logits.argmax(dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
        else:
            logits = logits.argmax(dim=-1).numpy()
            labels = labels.numpy()

        for i in range(0, len(logits)):
            if logits[i] == labels[i]:
                right_pred += 1

    print(right_pred, len(test_loader.dataset))
    accuracy = right_pred / len(test_loader.dataset)
    end_time = time.time()
    print('Testing Time: %d s, Accuracy: %f' % ((end_time - start_time), accuracy))


if __name__ == '__main__':
    ckpt = 'models/080.ckpt'
    batch_size = 64

    model = Resnet_50()
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    print(f"Model loaded from {ckpt}")

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        model = nn.DataParallel(model)

    test_loader = get_test_loader(batch_size)
    test(model, test_loader)