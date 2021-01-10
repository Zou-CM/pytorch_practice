#encoding=utf-8

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from dataset.MyDataset import MyDataset
from torchvision import models
import utils
from config import Config


def train(flag=True):
    cfg = Config()

    if flag:
        net = models.resnet101(pretrained=True)
        # net.fc = nn.Linear(2048, cfg.num_class)
        # net.aux_logits=False
        net.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, cfg.num_class)
        )
    else:
        net = models.resnet101(pretrained=False)
        net.fc = nn.Linear(2048, cfg.num_class)
        net.load_state_dict(torch.load(cfg.checkpoints_path))

    net.cuda()

    trainSet = MyDataset('train')
    devSet = MyDataset('dev')

    trainLoader = DataLoader(trainSet, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    devLoader = DataLoader(devSet, batch_size=1, shuffle=True, drop_last=True)

    for name, value in net.named_parameters():
        if 'fc' not in name:
            value.requires_grad = False

    params = filter(lambda p: p.requires_grad, net.parameters())

    opt = optim.Adam(params, lr=cfg.lr)

    ntr = len(trainLoader)

    for e in range(1, cfg.epoch+1):
        net.train()
        for n, data in enumerate(trainLoader):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            opt.zero_grad()
            ls = torch.nn.CrossEntropyLoss()(outputs, labels)
            ls.backward()
            opt.step()
            utils.bar(n, ntr, e, ls)
        net.eval()
        num = 0.0
        acc = 0.0
        for n, data in enumerate(devLoader):
            inputs, labels = data
            inputs = Variable(inputs)
            inputs = inputs.cuda()
            labels = Variable(labels)
            labels = labels.cuda()
            outputs = net(inputs)
            logits = torch.argmax(outputs, dim=1)
            for i in range(len(labels)):
                if labels[i] == logits[i]:
                    acc += 1
            num += 1.0
        print('\nEpoch %d acc = %.5f'%(e, acc / num))
        torch.save(net.state_dict(), cfg.checkpoints_path)
        if e != 0 and e % 5 == 0:
            for p in opt.param_groups:
                p['lr'] *= 0.5


def test():
    pass

if __name__ == '__main__':
    # net = models.inception_v3(False)
    # print(net)
    train(True)