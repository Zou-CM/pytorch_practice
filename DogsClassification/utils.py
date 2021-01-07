#encoding=utf-8

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def getDev():
    dev_path = './dev'
    train_path = './train'
    if not os.path.exists(dev_path):
        os.mkdir(dev_path)
    img_list = os.listdir(train_path)
    for i in range(len(img_list)):
        if i % 20 == 0:
            os.rename(os.path.join(train_path, img_list[i]), os.path.join(dev_path, img_list[i]))

def getLabel():
    file_path = './sample_submission.csv'
    names = []
    with open(file_path, 'r') as f:
        for line in f:
            print(line)
            names = line.strip().split(',')[1:]
            break
    labels = {}
    for i in range(len(names)):
        labels[names[i]] = i
    return labels

def renamePic():
    Map = {}
    with open('labels.csv', 'r') as f:
        for line in f:
            info = line.split(',')
            if info[0] == 'id':
                continue
            Map[info[0]] = info[1].strip()
    dev_path = './dev'
    train_path = './train'
    train_list = os.listdir(train_path)
    num = 0
    for item in train_list:
        name = item.split('.')
        os.rename(os.path.join(train_path, item), os.path.join(train_path, Map[name[0]] + '+' + str(num) + '.' + name[1]))
        num += 1
    dev_list = os.listdir(dev_path)
    for item in dev_list:
        name = item.split('.')
        os.rename(os.path.join(dev_path, item), os.path.join(dev_path, Map[name[0]] + '+' + str(num) + '.' + name[1]))
        num += 1

def show(img):
    img = (img + 1.0) / 2.0
    nimg = img.numpy()
    plt.imshow(np.transpose(nimg, (1, 2, 0)))
    plt.show()

def bar(t, s, e, ls):
    sys.stdout.write('\rEpoch ' + str(e) + ' [' + '*' * int(t*100/s+1) + ' ' * (100-int(t*100/s+1)) + '] ' +
                     'loss = %.5f' % ls)
    sys.stdout.flush()


if __name__ == '__main__':
    pass
    # getDev()
    # print(getLabel())
    # renamePic()