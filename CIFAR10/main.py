#encoding=utf-8

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import optim
from models.Net import Net
from data.MyDataset import MyDataset
from utils.show_tools import bar
from config import Config

def run():
    cfg = Config()

    trainSet = MyDataset('./source/pics/train/')
    testSet = MyDataset('./source/pics/test/')

    trainLoader = DataLoader(trainSet, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    testLoader = DataLoader(testSet, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    myNet = Net()
    opt = optim.Adam(myNet.parameters(), lr=cfg.learning_rate)

    ntr = len(trainLoader)
    nte = len(testLoader)

    for i in range(1, cfg.epoch+1):
        for n, data in enumerate(trainLoader):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)
            opt.zero_grad()
            outputs = myNet(inputs)
            ls = torch.nn.CrossEntropyLoss()(outputs, labels)
            ls.backward()
            opt.step()
            # print(n, ntr)
            bar(n, ntr)
        acc = 0
        num = 0
        for n, data in enumerate(testLoader):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = myNet(inputs)
            num += cfg.batch_size * 1.0
            logit = torch.argmax(outputs, dim=1)
            for index in range(16):
                if labels[index] == logit[index]:
                    acc += 1.0
        print('Epoch %d acc = %.3f'%(i, acc/num))

    return



if __name__ == '__main__':
    run()