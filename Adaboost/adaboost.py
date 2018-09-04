# -*- coding: utf-8 -*-

'''
根据P140 例子8.1来的数据
'''
import time
import numpy as np

'''
阈值分类器，有两种方向：
    1. x > v y = 1  x < v y = -1
    2. x > v y = -1 x < v y = 1
    v 是阈值
'''
class Sign(object):
    def __init__(self, features, labels, w):
        self.X = features               #x训练特征
        self.Y = labels                 #特征标签
        self.N = len(features)          #训练数据大小

        self.w = w                      #数据权值
        self.yCounts = [-1, 1]          #标签容器

    '''
    上面两种方向的第一种,
    1. x > v y = 1  x < v y = -1
    '''
    def trainLessthan(self):
        index = -1


