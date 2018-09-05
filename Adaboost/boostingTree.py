# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../DecisionTree/')
import leastSquaresRegressionTree

class BoostingTree(object):
    def __init__(self, trainX, trainY, maxIter):
        self.TX = trainX
        self.TY = trainY
        self.LSRTrees = []
        self.maxIter = maxIter

    '''
    生成一个回归树
    '''
    def newLSRTree(self, X, Y):
        tree = leastSquaresRegressionTree.LSRTree(X, Y, range(len(X)))                  #新建一个递归树
        tree.growTree()
        print tree.s, tree.err, tree.left.c, tree.right.c
        self.LSRTrees.append(tree)
    '''
    训练数据
    '''
    def train(self):
        currentY = self.TY
        for i in range(self.maxIter):
            self.newLSRTree(self.TX, currentY)                                      #根据当前的数据计算回归树
            if self.LSRTrees[-1].err < 0.2:
                break
            for j in range(len(self.TX)):                                           #计算残差
                predicty = self.LSRTrees[-1].predict(self.TX[j])                    #计算预测值
                ri = currentY[j] - predicty                                         #计算残差
                currentY[j] = ri                                                    #赋给当前的y值做下一次的回归树计算


if __name__ == '__main__':
    tx = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    ty = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    bt = BoostingTree(tx, ty, 10)
    bt.train()
