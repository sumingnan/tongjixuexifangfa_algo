# -*- coding: utf-8 -*-
#import pandas as pd
import numpy as np

import time
import math
import random

from collections import defaultdict


'''
最大熵模型
'''
class MaxEntropy(object):

    def init_params(self, trainX, trainY):
        self.TX = trainX
        self.TY = trainY
        self.Y_ = set()

        self.calPxyPx()                         #统计x和(x,y)在数据集中出现的次数

        self.N = len(trainX)                    #训练集大小
        self.n = len(self.Pxy)                  #特征函数的数目
        #self.M = 100000.0                       #所有特征在(x,y)出现的次数
        self.M = max([len(feature[0:]) for feature in self.TX])

        self.build_dict()
        self.calEpxy()                          #计算经验分布的期望值

    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}

        for i, (x, y) in enumerate(self.Pxy):
            self.id2xy[i] = (x, y)
            self.xy2id[(x, y)] = i
    '''
    统计联合分布的p(x,y)和p(x)的经验分布
    '''
    def calPxyPx(self):
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)

        for i in xrange(len(self.TX)):
            x_, y = self.TX[i], self.TY[i]
            self.Y_.add(y)

            for x in x_:
                self.Pxy[(x, y)] += 1
                self.Px[x] += 1

    '''
    检查(x,y)是否在特征函数中
    '''
    def fxy(self, x, y):
        return (x, y) in self.xy2id

    '''
    计算特征函数f(x,y)关于经验分布~p(x,y)的期望值（从数据集中计算）
    p82 书中的期望值
    '''
    def calEpxy(self):
        self.Epxy = defaultdict(float)
        for id in xrange(self.n):           #遍历特征函数中的数据
            (x, y) = self.id2xy[id]
            self.Epxy[id] = float(self.Pxy[(x,y)]) / float(self.N)  * 1.0       #计算特征函数(x,y)的期望值


    '''
    计算(x,y)的条件熵
    '''
    def calPxy(self, X, y):
        result = 0.0
        for x in X:
            if self.fxy(x, y):
                id = self.xy2id[(x,y)]
                result += self.w[id] * 1.0

        return (math.exp(result), y)

    '''
    计算条件熵，书中公式6.22
    '''
    def calProbalityYX(self, X):
        Pyxs = [(self.calPxy(X, y)) for y in self.Y_]   #计算每一(x,y)的值
        z = sum([prob for prob, y in Pyxs])             #计算zw的值
        return [(prob / z, y) for prob, y in Pyxs]
    '''
    计算模型分布的期望值，书中是83页
    '''
    def calEpx(self):
        self.Epx = [0.0 for i in xrange(self.n)]          #初始化

        for i, X in enumerate(self.TX):
            Pyxs = self.calProbalityYX(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x,y)]
                        self.Epx[id] += Pyx * (1.0 / self.N) * 1.0      #1.0是特征函数的值，计算期望值

    '''
    训练数据
    '''
    def train(self, TX, TY):
        self.init_params(TX, TY)                    #初始化数据，并计算经验分布的期望值
        self.w = [0.0 for i in range(self.n)]       #初始化期望值

        maxIter = 1000
        for time in xrange(maxIter):
            alphas = []
            self.calEpx()

            for i in xrange(self.n):
                alpha = 1.0 / self.M * math.log(self.Epxy[i] / self.Epx[i])
                alphas.append(alpha)

            self.w = [self.w[i] + alphas[i] for i in xrange(self.n)]

    '''
    预测
    '''
    def predict(self, testSet):
        results = []
        for test in testSet:
            result = self.calProbalityYX(test)
            results.append(result)
            #results.append(max(result,key=lambda x: x[0])[1])
        return results

if __name__ == "__main__":
    trainX = []
    trainY = []
    for line in open("testData.dat"):
        fields = line.strip().split()
        trainY.append(fields[0])
        trainX.append(fields[1:])

    ent = MaxEntropy()
    ent.train(trainX, trainY)
    print ent.predict([['Sunny']])
