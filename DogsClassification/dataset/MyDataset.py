#encoding=utf-8

from torch.utils import data
from torchvision import transforms
import os
from PIL import Image
from DogsClassification import utils

class MyDataset(data.Dataset):
    def __init__(self, root):
        self.root = './' + root
        self.img_list = os.listdir(self.root)
        self.Map = utils.getLabel()
        self.transforms = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomRotation((-90, 90)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        path = os.path.join(self.root, self.img_list[index])
        pic = Image.open(path)
        pic = self.transforms(pic)
        label = ""
        if self.root != './test':
            label = self.Map[self.img_list[index].split('+')[0]]
        return pic, label

    def __len__(self):
        return len(self.img_list)


