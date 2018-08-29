# -*- coding: utf-8 -*-
#logisticRegression算法
from numpy import *
import matplotlib.pyplot as plt
import time

'''
计算激活函数
'''
def sigmoid(z):
    return 1.0 / (1 + exp(-z))

'''
逻辑斯蒂算法的训练，这里面有梯度算法、随机梯度算法和改进的随机梯度算法，供比较
'''
def trainLogisticRegression(trainX, trainY, opt):
    beginTime = time.time()                                     #计时开始

    numSampers, numFeatures = shape(trainX)                     #返回样本数目和特征向量数
    alpha = opt["alpha"]                                        #学习率
    maxIter = opt["maxIter"]                                    #最大的迭代数目
    weights = ones((numFeatures, 1))                            #初始权重

    for k in range(maxIter):
        if opt["optimizeType"] == "gradDescent":                #梯度算法
            output = sigmoid(trainX * weights)
            error = trainY - output
            weights = weights + alpha * trainX.transpose() * error
        elif opt["optimizeType"] == "stocGradDescent":          #随机梯度算法
            for i in range(numSampers):
                output = sigmoid(trainX[i,:] * weights)
                error = trainY[i, 0] - output
                weights = weights + alpha * trainX[i,:].transpose() * error
        elif opt["optimizeType"] == "smoothStocGradDescent":    #改进的随机梯度算法
            dataIndex = range(numSampers)
            for i in range(numSampers):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(trainX[randIndex,:] * weights)
                error = trainY[randIndex, 0] - output
                weights = weights + alpha * trainX[randIndex,:].transpose() * error
                del(dataIndex[randIndex])
        else:
            raise NameError("Not support optimiz method type!")

    print 'Congratulations, training complete! Take %fs!' %(time.time() - beginTime)
    return weights

'''
测试正确率
'''
def calLogisticRegressions(weights, testX, testY):
    numSamples, numFeatures = shape(testX)
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(testX[i,:] * weights)[0, 0] > 0.5
        if predict == bool(testY[i, 0]):
            matchCount += 1

    accuracy = float(matchCount) / numSamples
    return accuracy

'''
显示训练出来的逻辑斯蹄算法的参数
'''
def showLogisticRegression(weights, trainX, trainY):
    numSamples, numFeatures = shape(trainX)
    if numFeatures != 3:
        print "i can not know how to draw"
        return 1

    for i in xrange(numSamples):
        if int(trainY[i, 0]) == 0:
            plt.plot(trainX[i, 1], trainX[i, 2], 'or')
        elif int(trainY[i, 0]) == 1:
            plt.plot(trainX[i, 1], trainX[i, 2], 'ob')

    minX = min(trainX[:,1])[0, 0]
    maxX = max(trainX[:,1])[0, 0]

    weights = weights.getA()
    yMinX = float(-weights[0] - weights[1] * minX) / weights[2]
    yMaxX = float(-weights[0] - weights[1] * maxX) / weights[2]

    plt.plot([minX, maxX], [yMinX, yMaxX], '-g')
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

'''
加载数据
'''
def loadDataSet():
    trainX = []
    trainY = []
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineAttr = line.strip().split()
        trainX.append([1.0, float(lineAttr[0]), float(lineAttr[1])])
        trainY.append(float(lineAttr[2]))
    return mat(trainX), mat(trainY).transpose()

if __name__ == '__main__':
    #1. 加载数据
    trainX, trainY = loadDataSet()
    testX = trainX
    testY = trainY

    #2 训练模型
    opt = {'alpha':0.01, 'maxIter':500, 'optimizeType':'gradDescent'}
    #opt = {'alpha': 0.01, 'maxIter': 200, 'optimizeType': 'stocGradDescent'}
    #opt = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
    optimalWeights = trainLogisticRegression(trainX, trainY, opt)

    #3.测试
    accuracy = calLogisticRegressions(optimalWeights, testX, testY)

    #4 显示图像
    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
    showLogisticRegression(optimalWeights, trainX, trainY)

