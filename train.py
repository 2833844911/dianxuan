
import torch.nn as nn
import torch
import torch.utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import random
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    x = "cuda"
    print('正在使用gpu训练')
else:
    x = 'cpu'
    print('正在使用cpu训练')

device = torch.device(x)

mymod = models.vgg16(pretrained=True)

del mymod.avgpool
del mymod.classifier

def getletbie(path):
    data = {}
    for i in os.listdir(path):
        data[i] = [path+'/'+i+'/'+ik for ik in os.listdir(path+'/'+i)]
    return data

def getrandom(data, lb, ko=0):
    keylist = lb.keys()
    j = list(keylist)
    j.remove(data)
    if ko == 1:
        f = data
    else:
        f = random.choice(j)
    return random.choice(lb[f])



def getsjj(data):
    alldata = []
    for i in data:
        for k in data[i]:
            ku = [[k, getrandom(i,data), 0], [k, getrandom(i,data, 1), 1]]
            random.shuffle(ku)
            alldata.append(ku)

    return alldata

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

class Datasjj(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.l =len(self.data)
        self.tpzq = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()

        ])

    def __getitem__(self, item):
        k = self.data[item]

        d = Image.open(k[0][0])
        d = self.tpzq(d)

        c = Image.open(k[0][1])
        c = self.tpzq(c)

        d2 = Image.open(k[1][0])
        d2 = self.tpzq(d2)

        c2 = Image.open(k[1][1])
        c2 = self.tpzq(c2)

        d = d.to(device).unsqueeze(0)
        c = c.to(device).unsqueeze(0)
        d2 = d2.to(device).unsqueeze(0)
        c2 = c2.to(device).unsqueeze(0)
        return torch.concat([d, d2], dim=0), torch.concat([c, c2], dim=0), torch.tensor([k[0][2], k[1][2]],dtype=torch.float).to(device)
    def __len__(self):
        return self.l

def train():
    mymox.train()
    allloss = 0
    idx = 0
    zql = 0

    mk = tqdm(traf)
    for k,x, t in mk:
        k = k.view(k.shape[0] * k.shape[1], k.shape[2], k.shape[3], k.shape[4])
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        t = t.view(t.shape[0]* t.shape[1], 1)
        Adme.zero_grad()

        out = mymox(k,x)
        ls = myloss(out, t)
        ls.backward()
        Adme.step()
        allloss += ls.item()
        with torch.no_grad():
            equal = torch.eq(torch.round(out), t)
            d = torch.mean(equal.float())

        zql += d.item()



        idx+=1
        mk.set_description(desc='loss [{}] zql [{}]'.format(allloss/idx, zql/idx))
    return allloss/idx




def getSjTrain():
    f = getletbie('./train')
    sjj = getsjj(f)
    tra = Datasjj(sjj)
    f = DataLoader(tra, shuffle=True, batch_size=20)

    return f


def getSjText():
    f = getletbie('./val')
    sjj = getsjj(f)
    tra = Datasjj(sjj)
    f = DataLoader(tra, shuffle=True, batch_size=10)

    return f

def text():
    mymox.eval()
    allloss = 0
    zql = 0
    idx = 0
    mk = tqdm(texf)
    with torch.no_grad():
        for k, x, t in mk:
            # t = t.view(t.shape[0], 1)

            k = k.view(k.shape[0] * k.shape[1], k.shape[2], k.shape[3], k.shape[4])
            x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
            t = t.view(t.shape[0]* t.shape[1], 1)
            out = mymox(k, x)
            ls = myloss(out, t)

            allloss += ls.item()
            idx += 1

            with torch.no_grad():
                equal = torch.eq(torch.round(out), t)
                d = torch.mean(equal.float())
            zql += d.item()

            mk.set_description(desc='loss [{}] zql [{}]'.format(allloss / idx, zql/idx))


if __name__ == '__main__':


    mymox = Siamese()  # 重新训练
    # mymox = torch.load('./bj.pth') # 迁移学习

    epoch = 10 # 训练多少epoch

    mymox.to(device)
    Adme = optim.Adam(mymox.parameters(),lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(Adme, step_size=5, gamma=0.1)
    myloss = nn.BCELoss()

    ls = 10000

    for i in range(epoch):
        print('epoch', i+1)
        traf = getSjTrain()
        texf = getSjText()
        f = train()

        text()
        scheduler.step()
        if f < ls:
            ls = f
            print('保存模型===>', 'bj.pth')
            torch.save(mymox,'bj.pth')





