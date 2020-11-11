#encoding=utf-8

import pickle
import numpy as np
import os
import cv2
import sys

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_file(dict):
    X = dict['data']  # X, ndarray, 像素值
    Y = dict['labels']  # Y, list, 标签, 分类
    N = dict['filenames']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y, N

def load_CIFAR10(root):
    xs = []
    ys = []
    ns = []
    # for i in range(1, 6):
    #     path = os.path.join(root, 'data_batch_%d'%i)
    #     # print(path)
    #     dict = unpickle(path)
    #     # print(dict)
    #     X, Y, N = load_file(dict)
    #     xs.extend(X)
    #     ys.extend(Y)
    #     ns.extend(N)
    path = os.path.join(root, 'test_batch')
    dict = unpickle(path)
    X, Y, N = load_file(dict)
    xs.extend(X)
    ys.extend(Y)
    ns.extend(N)
    return xs, ys, ns



if __name__ == '__main__':
    xs, ys, ns = load_CIFAR10('../source/cifar-10-batches-py')
    xs = np.array(xs)
    ys = np.array(ys)
    ns = np.array(ns)
    print(np.shape(xs))
    for i in range(len(xs)):
        img = xs[i, :, :, :]
        # print(type(img))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite('../source/pics/test/' + str(ys[i]) + '_' + str(i) + '.png', img)
        sys.stdout.write('\r[' + '*'*int(i*100/len(xs)+1) + ' '*(99 - int(i*100/len(xs))) + ']' + '%d'%int(i*100/len(xs)+1) + '%')
        sys.stdout.flush()