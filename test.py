# 测试模型，生成结果，计算准确率
import torch
from PIL import Image
from tqdm.auto import tqdm
from data import test_dataloader
def test(**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    predictions=[]

    for batch in tqdm(get_test_loader):
        imgs, label=batch

        with torch.no_grad():
            logits=model(imgs.to(device))

        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    with open("predict.csv","w") as f:


        f.write("Id,Category\n")

        for i,pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")