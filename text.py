import torch.nn as nn
import torch
import torch.utils
from PIL import Image
import os
import torchvision.models as models
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    x = "cuda"
    print('正在使用gpu识别')
else:
    x = 'cpu'
    print('正在使用cpu识别')
device = torch.device(x)

mymod = models.vgg16(pretrained=True)

del mymod.avgpool
del mymod.classifier


class Siamese(nn.Module):
    def __init__(self, pretrained=True):
        super(Siamese, self).__init__()
        self.resnet = mymod.features
        self.resnet = self.resnet.eval()
        self.resnet.to(device)
        flat_shape = 512 * 3 * 3
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)
        self.sgm = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)


        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        x = self.sgm(x)
        return x

def getdata(p1, p2):
    p1 = Image.open(p1)
    ch = tpzq(p1)
    ch =  ch.to(device).unsqueeze(0)
    p2 = Image.open(p2)
    ch2 = tpzq(p2)
    ch2 = ch2.to(device).unsqueeze(0)
    h = mymox(ch, ch2)
    return h[0,0].item()
tpzq = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor()])
if __name__ == '__main__':


    mymox = torch.load('./bj.pth')
    mymox.to(device)

    xsd = getdata(r'D:\数据集\点选训练集\val\丁\丁_672.png', r'D:\数据集\点选训练集\val\七\七_28528.png')
    print("图片相似度为",xsd)







