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
class Adaboost(object):
    def __init__(self, trainX, trainY, maxIter=100):
        self.X = trainX
        self.Y = trainY
        self.N = len(trainX)                                #训练数目
        self.n = len(trainX[0])                             #特征维度
        self.weakClass = []                                 #弱分类器列表
        self.D = np.mat(np.ones((self.N, 1))/ self.N)       #初始化训练数据的权值分布
        self.maxIter = maxIter                              #弱分类器数目上限

    '''
    某一特征值阈值分类
    '''
    def stumpClassify(self, dataMatrix, dim, thresholdV, thresholdIneq):
        returnValues = np.ones((np.shape(dataMatrix)[0], 1))

        if thresholdIneq == 'lt':
            returnValues[dataMatrix[:,dim] <= thresholdV] = -1
        else:
            returnValues[dataMatrix[:,dim] > thresholdV] = -1

        return returnValues
    '''
    新建弱分类器
    '''
    def newWeakClassify(self):
        trainX = np.mat(self.X)
        trainY = np.mat(self.Y).T

        stepNum = 10.0
        bestClassify = {}
        bestClassPredict = np.mat(np.zeros((self.N, 1)))
        minError = np.inf
        for i in range(self.n):             #遍历特征向量
            rangeMin = trainX[:,i].min()
            rangeMax = trainX[:,i].max()
            stepSize = (rangeMax - rangeMin) / stepNum
            for j in range(-1, int(stepNum) + 1):
                for thresholdIneq in ['lt', 'gt']:
                    thresholdValue = rangeMin + float(j) * stepSize
                    predictClass = self.stumpClassify(self.X, i, thresholdValue, thresholdIneq)
                    errorArray = np.mat(np.ones((self.N, 1)))
                    errorArray[predictClass == trainY] = 0                      #统计误差样本
                    e_m = self.D.T * errorArray                                 #计算分类误差率
                    if e_m < minError:
                        minError = e_m
                        bestClassPredict = predictClass.copy()
                        bestClassify['dim'] = i
                        bestClassify['thresholdValue'] = thresholdValue
                        bestClassify['thresholdIneq'] = thresholdIneq

        return bestClassPredict, minError, bestClassify

    '''
    训练数据
    '''
    def trainAdaboost(self):
        fxPredict = np.mat(np.zeros((self.N, 1)))
        for i in range(self.maxIter):
            bestClassPredict, minError, bestClassify = self.newWeakClassify()       #查找分类误差率最低的基本分类器
            alpha_m = self.calAlpha(minError)                                       #计算alpha值
            bestClassify['alpha_m'] = alpha_m
            self.weakClass.append(bestClassify)                                     #添加到基本分类器的队列中
            self.calD(alpha_m, bestClassPredict)                                    #更新训练集的权重分布
            print minError
            fxPredict += alpha_m * bestClassPredict
            fxError = np.multiply(np.sign(fxPredict) != np.mat(self.Y).T, np.ones((self.N, 1)))
            fxErrorRate = fxError.sum() / self.N
            if fxErrorRate == 0.0:                                                  #无误差了
                break
        print self.weakClass
    '''
    计算alpha值，对应公式中的8.2
    '''
    def calAlpha(self, e_m):
        return float(0.5 * np.log((1 - e_m) / max(e_m, 1e-16)))

    '''
    计算每一轮的权值分布值，公式8.3、8.4、8.5
    '''
    def calD(self, alpha, bestClassPredict):
        exposn = np.multiply(-1.0 * alpha * np.mat(self.Y).T, bestClassPredict)
        wi = np.multiply(self.D, np.exp(exposn))        #计算wi值
        zm = wi.sum()                                   #计算zm值
        self.D = wi / zm                                #计算训练集的权重

    '''
    预测函数
    '''
    def adaboostPredict(self, testX):
        testMaxtirxX = np.mat(testX)
        m = np.shape(testMaxtirxX)[0]                   #待预测的数目
        predicts = np.mat(np.zeros((m, 1)))             #预测结果
        for i in range(len(self.weakClass)):            #遍历所有基本分类器
            predict = self.stumpClassify(testX, self.weakClass[i]['dim'], self.weakClass[i]['thresholdValue'], self.weakClass[i]['thresholdIneq'])
            predicts += self.weakClass[i]['alpha_m'] * predict
        return np.sign(predicts)

if __name__ == "__main__":
    datMat = np.matrix([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    classLabels = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
    adaboost = Adaboost(datMat, classLabels)
    adaboost.trainAdaboost()

