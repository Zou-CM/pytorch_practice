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
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomRotation((-30, 30)),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        path = os.path.join(self.root, self.img_list[index])
        pic = Image.open(path)
        # pic = pic.resize((300, 300))
        pic = self.transforms(pic)
        label = self.Map[self.img_list[index].split('+')[0]]
        return pic, label

    def __len__(self):
        return len(self.img_list)


