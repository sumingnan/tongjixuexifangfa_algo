#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 感知机对偶模式
# 李航  统计学习方法

import numpy as np
import random

class Perceptron(object):
    def __init__(self, learn):
        self.learningStep = learn   #学习率

    def sigmod(self, x):
        return (x >= 0.0)

    def printWeight(self):
        print "f(x) = sign(" , self.weight[0] , "x1" , " + " , self.weight[1] , "x2" , " + " , self.bias , ")"

    def dot(self, x, y):
        result = 0.0
        for i in xrange(len(x)):
            result += x[i] * y[i]
        return result

    def gram(self, x):#计算gram矩阵
        result = []
        for i in xrange(len(x)):
            row = []
            currRow = list(x[i])
            for j in xrange(len(x)):
                currColum = list(x[j])
                row.append(self.dot(currRow, currColum))
            result.append(row)
        return result

    def train(self, x, y):      #训练
        self.weight = [0.0] * len(x[0])
        self.bias = 0.0
        alpha = [0.0] * len(x)  #选取alpha值
        b = 0.0

        gramValue = self.gram(x)
        index = 0
        featuresNum = len(x)
        while index < featuresNum:
            currentFeature = list(x[index]) #在训练集中选取数据

            wx = 0.0
            currRow = list(gramValue[index])
            for i in xrange(len(x)):
                wx += alpha[i] * y[i] * currRow[i]
            wx += b

            if wx * y[index] <= 0:  # 如果，误分类，需更新alpha值
                alpha[index] += self.learningStep
                b += self.learningStep * y[index]
                index = 0
            else:
                index += 1
        wx = 0.0
        for i in xrange(len(x)):#计算权重值
            x_temp = alpha[i] * y[i]
            for j in xrange(len(x[0])):
                self.weight[j] += x_temp * x[i][j]
        self.bias = b

    def predict(self, x):#预测
        feature = list(x)
        wx = 0.0
        for i in xrange(len(feature)):
            wx += self.weight[i] * feature[i]
        wx += self.bias

        return self.sigmod(wx)

if __name__ == '__main__':
    p = Perceptron(1)
    x = [[3, 3], [4, 3], [1, 1]]
    y = [1.0, 1.0, -1.0]
    p.train(x, y)
    p.printWeight()
    print p.predict([4, 4])