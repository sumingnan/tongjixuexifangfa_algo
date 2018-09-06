# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../DecisionTree/')
import leastSquaresRegressionTree

'''
梯度值，不等式是左开右闭  0<x<=2.5
'''
class GradientValue(object):
    def __init__(self, left, right, value):
        self.value = value
        self.left = left
        self.right = right

class BoostingTree(object):
    def __init__(self, trainX, trainY, maxIter):
        self.TX = trainX
        self.TY = trainY
        self.LSRTrees = []
        self.maxIter = maxIter
        self.fx = []

    '''
    生成一个回归树
    '''
    def newLSRTree(self, X, Y):
        tree = leastSquaresRegressionTree.LSRTree(X, Y, range(len(X)))                  #新建一个递归树
        tree.growTree()
        #print tree.s, tree.err, tree.left.c, tree.right.c
        self.LSRTrees.append(tree)

    '''
    计算f(x) + t(x)
    '''
    def calFT(self, tree):
        if len(self.fx) == 0:
            self.fx.append( GradientValue(-np.inf, tree.s, tree.left.c))
            self.fx.append(GradientValue(tree.s, np.inf, tree.right.c))
        else:
            inserIndex = -1
            updateIndex = -1
            for i in range(len(self.fx)):
                if tree.s < self.fx[i].left:
                    inserIndex = i-1
                    break
                elif tree.s == self.fx[i].left:
                    updateIndex = i-1
                    break
                elif tree.s < self.fx[i].right:
                    inserIndex = i
                    break
                elif tree.s == self.fx[i].right:
                    updateIndex = i
                    break
            if inserIndex > -1:                 #待插入数据中
                for i in range(0, inserIndex):
                    self.fx[i].value += tree.left.c
                for i in range(inserIndex+1, len(self.fx)):
                    self.fx[i].value += tree.right.c
                originLeft = self.fx[inserIndex].left
                originValue = self.fx[inserIndex].value
                self.fx[inserIndex].left = tree.s
                self.fx[inserIndex].value = originValue + tree.right.c
                self.fx.insert(inserIndex, GradientValue(originLeft, tree.s, originValue + tree.left.c))
            if updateIndex > -1:
                for i in range(0, updateIndex+1):
                    self.fx[i].value += tree.left.c
                for i in range(updateIndex + 1, len(self.fx)):
                    self.fx[i].value += tree.right.c
    '''
    训练数据
    '''
    def train(self):
        currentY = self.TY.copy()
        for i in range(self.maxIter):
            self.newLSRTree(self.TX, currentY)                                      #根据当前的数据计算回归树
            self.calFT(self.LSRTrees[-1])                                           #计算Fx + Tx
            if self.LSRTrees[-1].err < 0.2:
                break
            for j in range(len(self.TX)):                                           #计算残差
                predicty = self._predict(self.TX[j])                                #计算预测值
                ri = self.TY[j] - predicty                                       #计算残差
                currentY[j] = ri                                                    #赋给当前的y值做下一次的回归树计算
    '''
    内部预测函数
    '''
    def _predict(self, x):
        for i in range(len(self.fx)):
            if x <= self.fx[i].right:
                return self.fx[i].value
        return None


if __name__ == '__main__':
    tx = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    ty = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    bt = BoostingTree(tx, ty, 10)
    bt.train()
    for i in range(len(bt.fx)):
        print bt.fx[i].left, bt.fx[i].right, bt.fx[i].value