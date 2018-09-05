# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
'''
最小二乘回归树 类
'''
class LSRTree(object):
    def __init__(self, data, y, sepIdx):
        self.data = data
        self.y = y
        self.sepIdx = sepIdx                            #当前结点所包含的顶点序列号
        self.left = None
        self.right = None
        self.j = None
        self.s = None
        self.c = np.mean(y[sepIdx])                     #当前节点的平均值（即为c值）
        self.isLeaf = True
        self.err = 0
    '''
    计算(j, s)的误差
    '''
    def calSquaError(self, data, y, sepIdx, j, s):
        left = []                                       #左子树
        right = []                                      #右子树
        for i in sepIdx:
            if data[i][j] <= s:
                left.append(i)
            else:
                right.append(i)

        leftMean = np.mean(y[left])                     #计算R1的平均值
        leftErr  = np.sum((y[left] - leftMean) ** 2)    #计算R1的平方误差

        rightMean = np.mean(y[right])
        rightErr  = np.sum((y[right] - rightMean) ** 2)

        return leftErr + rightErr
    '''
    寻找最佳的(j, s)对使误差最小
    '''
    def findBestSegmentation(self, data, y, sepIdx):
        bestJ = 0                                       #选取j的初始值
        sortedXValue = np.sort(data[sepIdx][:,bestJ])   #对特征j的取值排序
        bestS = (sortedXValue[0] + sortedXValue[1]) / 2 #特征值的初始值
        bestErr = self.calSquaError(data, y, sepIdx, bestJ, bestS)

        for j in range(data.shape[1]):                  #遍历所有的特征向量
            currSortedXValue = np.sort(data[sepIdx][:,j])
            sValues = (currSortedXValue[1:] + currSortedXValue[:-1]) / 2    #(1+2/2, 2+3/2,....)切分值用平均值处理
            for s in sValues:
                currError = self.calSquaError(data, y, sepIdx, j, s)
                if currError < bestErr:
                    bestErr = currError
                    bestJ = j
                    bestS = s
        return bestJ, bestS, bestErr
    '''
    对(j, s)划分区域，并生成左右子树
    '''
    def cart(self):
        if len(self.sepIdx):
            j, s, error = self.findBestSegmentation(self.data, self.y, self.sepIdx)
            leftIdx, rightIdx = [], []
            for i in self.sepIdx:
                if self.data[i][j] <= s:
                    leftIdx.append(i)
                else:
                    rightIdx.append(i)
            self.isLeaf = False
            self.left = LSRTree(self.data, self.y, leftIdx)     #建立左子树
            self.right = LSRTree(self.data, self.y, rightIdx)   #建立右子树
            self.j = j
            self.s = s
            self.err = error
    '''
    计算此子树的误差
    '''
    def error(self):
        return np.sum((self.y[self.sepIdx] - self.c) ** 2)

    '''
    预测数据的输出
    '''
    def predict(self, point):
        tree = self
        while True:
            if tree.isLeaf:
                return tree.c
            else:
                if point[tree.j] <= tree.s:
                    tree = tree.left
                else:
                    tree = tree.right

    '''
    继续划分左右子树
    '''
    def growTree(self):
        if self.isLeaf:
            self.cart()
        else:
            self.left.growTree()
            self.right.growTree()

if __name__ == '__main__':
    testDataNum = 500
    datas = np.random.rand(testDataNum, 2)
    y = datas[:,0] + datas[:,1] + 0.2 * (np.random.rand(testDataNum) - 0.5)

    rootTree = LSRTree(datas, y, range(testDataNum))
    for i in range(5):
        rootTree.growTree()
        y_predict = np.array([rootTree.predict(p) for p in datas])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(datas[:,0], datas[:,1], y)
        ax.scatter(datas[:,0], datas[:,1], y_predict)
        plt.show()