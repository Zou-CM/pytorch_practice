#encoding=utf-8

import sys

def bar(t, s):
    """
    手写的一个进度条函数，不然都不知道训练的进度，很难受：）
    :param t: 当前运行的数量
    :param s: 总数量
    :return:
    """
    sys.stdout.write('\r[' + '*'*int(t*100/s+1) + ' '*(99 - int(t*100/s)) + ']' + '%d'%int(t*100/s+1) + '%')
    sys.stdout.flush()