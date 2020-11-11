#encoding=utf-8

from torch.utils import data
from torchvision import transforms
import os
import cv2

class MyDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.path = os.listdir(root)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        path = os.path.join(self.root, self.path[index])
        pic = cv2.imread(path)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic = self.transforms(pic)
        label = int(self.path[index].split('_')[0])
        return pic, label

    def __len__(self):
        return len(self.path)
