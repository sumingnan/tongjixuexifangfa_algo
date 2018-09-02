# -*- coding: utf-8 -*-
'''
SVM算法中的smo算法实现，参考网上
'''
import time
import random
import logging

import pandas as pd



class SVM(object):
    '''
    构造函数
    '''
    def __init__(self, kernel = 'linear', epsion = 0.01):
        self.kernel = kernel
        self.epsilon = epsion

    '''
    初始化参数
    '''
    def initParameters(self, features, labels):
        self.X = features
        self.Y = labels

        self.b = 0.0
        self.n = len(features[0])                       #特征数目
        self.N = len(features)                          #训练样本数目
        self.alpha = [0.0] * self.N
        self.E = [self.calE[i] for i in xrange(self.N)]    #计算差值

        self.C = 1000
        self.MaxIteration = 5000                        #最大迭代数目

    '''
    计算差值E，P127 公式7.105
    '''
    def calE(self, i):
        return self.G(i) - self.Y[i]

    '''
    计算g值，P127 公式7.104
    '''
    def G(self, i):
        result = self.b

        for j in xrange(self.N):
            result += self.alpha[j] * self.Y[j] * self.K(self.X[i], self.X[j])

        return result

    '''
    计算内核值，核函数
    '''
    def K(self, x1, x2):

        if self.kernel == 'linear':
            return sum([x1[k] * x2[k] for k in xrange(self.n)])
        if self.kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in xrange(self.n)]) + 1) ** 3

        print '没有定义核函数'
        return 0

    '''
    判断是否满足KKT条件，P130
    '''
    def satisfyKKT(self, i):
        ygx = self.Y[i] * self.G(i)                     #计算y*g
        if abs(self.alpha[i]) < self.epsilon:           #>=1
            return ygx > 1 or ygx == 1
        elif abs(self.alpha[i] - self.C) < self.epsilon:
            return ygx < 1 or ygx == 1
        else:
            return abs(ygx-1) < self.epsilon

    '''
    判断是否可以停止
    '''
    def isStop(self):
        for i in xrange(self.N):
            satisfy = self.satisfyKKT(i)

            if not satisfy:
                return False
        return True

    '''
    按照《统计学习方法》上的P129上的方法选取两个变量
    '''
    def selectTwoParameters(self):
        indexist = [i for i in xrange(self.N)]

        ilList1 = filter(lambda i: self.alpha[i] > 0 and self.alpha[i] < self.C, indexist)
        ilList2 = list(set(indexist) - set(ilList1))

        ilList = ilList1
        ilList.extend(ilList2)

        for i in ilList:
            if self.satisfyKKT(i):
                continue

            E1 = self.E[i]
            max_ = (0, 0)

            for j in indexist:
                if i == j:
                    continue

                E2 = self.E[j]
                if abs(E1 - E2) > max_[0]:
                    max_ = (abs(E1 - E2), j)

            return i, max_[1]

    '''
    训练数据
    '''
    def train(self, features, labels):

        self.initParameters(features, labels)

        for times in xrange(self.MaxIteration):

            i1, i2 = self.selectTwoParameters()                         #选取两个变量

            L = max(0, self.alpha[i2] - self.alpha[i1])                 #P126的计算方法，默认是y1 * y2 = -1
            H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            if self.Y[i1] == self.Y[i2]:                                #P126的计算方法，y1 * y2 = 1
                L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
                H = min(self.C, self.alpha[i2] + self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            #P127 公式7.107
            eta = self.K(self.X[i1], self.X[i1]) + self.K(self.X[i2], self.X[i2]) - 2 * self.K(self.X[i1], self.X[i2])

            alpha2NewUnc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta    #P127 公式7.106

            #P127 公式7.108
            alpha2New = 0
            if alpha2NewUnc > H:
                alpha2New = H
            elif alpha2NewUnc < L:
                alpha2New = L
            else:
                alpha2New = alpha2NewUnc

            #P127 公式7.109
            alpha1New = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2New)

            #P130 公式7.115计算
            bNew = 0
            b1New = -E1 - self.Y[i1] * self.K(self.X[i1], self.X[i1]) * (alpha1New - self.alpha[i1]) - \
                    self.Y[i2] * self.K(self.X[i2], self.X[i1]) * (alpha2New - self.alpha[i2]) + self.b
            #P130 公式7.116
            b2New = -E2 - self.Y[i1] * self.K(self.X[i1], self.X[i2]) * (alpha1New - self.alpha[i1]) - self.Y[i2] * self.K(self.X[i2], self.X[i2]) * (alpha2New - self.alpha[i2]) + self.b

            if alpha1New > 0 and alpha1New < self.C:
                bNew = b1New
            elif alpha2New > 0 and alpha2New < self.C:
                bNew = b2New
            else:
                bNew = (b1New + b2New) / 2

            self.alpha[i1] = alpha1New
            self.alpha[i2] = alpha2New
            self.b = bNew

            self.E[i1] = self.calE(i1)
            self.E[i2] = self.calE(i2)

    '''
    预测函数
    '''
    def _predict(self, feature):
        result = self.b

        for i in xrange(self.N):
            result += self.alpha[i] * self.Y[i] * self.K(feature, self.X[i])

        if result > 0:
            return 1
        return -1

    def predict(self, features):
        results = []

        for feature in features:
            results.append(self._predict(feature))