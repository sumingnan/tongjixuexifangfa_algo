#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 感知机原始模式
# 李航  统计学习方法

import numpy as np
import random

class Perceptron(object):
    def __init__(self, learn):
        self.learningStep = learn   #学习率

    def sigmod(self, x):
        return (x >= 0.0)

    def printWeight(self):
        print "f(x) = sign(" , self.weight[0] , "x1" , " + " , self.weight[1] , "x2" , " + " , self.weight[2] , ")"

    def train(self, x, y):
        self.weight = [0.0] * (len(x[0]) + 1)  #选取初始值

        index = 0
        featuresNum = len(x)
        while index < featuresNum:
            currentFeature = list(x[index]) #在训练集中选取数据
            currentFeature.append(1.0)
            wx = 0.0
            for j in xrange(len(self.weight)):
                wx += self.weight[j] * currentFeature[j]

            if wx * y[index] <= 0:  # 如果，误分类，需更新权重向量
                for j in xrange(len(self.weight)):
                    self.weight[j] += self.learningStep * (y[index] * currentFeature[j])
                index = 0
            else:
                index += 1

    def predict(self, x):
        feature = list(x)
        feature.append(1)
        wx = 0.0
        for i in xrange(len(feature)):
            wx += self.weight[i] * feature[i]

        return self.sigmod(wx)

if __name__ == '__main__':
    p = Perceptron(1)
    x = [[3, 3], [4, 3], [1, 1]]
    y = [1.0, 1.0, -1.0]
    p.train(x, y)
    p.printWeight()
    print p.predict([4, 4])