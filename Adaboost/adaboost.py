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
        bestV = 100000
        minError = 1000000
        for i in xrange(self.N):
            v = self.X[i]
            error = 0
            for j in xrange(self.N):
                if self.X[j] < v:
                    if self.Y[j] == 1:
                        error = error + self.w[j]
                else:
                    if self.Y[j] == -1:
                        error = error + self.w[j]
            if error < minError:
                bestV = v
                minError = error

    '''
    上面两种方向的第二种,
    2. x > v y = -1 x < v y = 1
    '''

    def trainLargethan(self):
        bestV = 100000
        minError = 1000000
        for i in xrange(self.N):
            v = self.X[i]
            error = 0
            for j in xrange(self.N):
                if self.X[j] > v:
                    if self.Y[j] == 1:
                        error = error + self.w[j]
                else:
                    if self.Y[j] == -1:
                        error = error + self.w[j]
            if error < minError:
                bestV = v
                minError = error

'''
adaboost算法，只针对一维数据，用来理解代码使用而已，并没有考虑效率等等
'''
class Adaboost(object):
    