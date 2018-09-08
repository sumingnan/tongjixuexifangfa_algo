# -*- coding: utf-8 -*-
import numpy as np

class GMM(object):
    def __init__(self, dataArray, threshold = 1e-6, maxStep = 20):
        self.dataArray = dataArray
        self.threshold = threshold
        self.maxSteps = maxStep

    '''
    计算高斯分布密度 公式9.25 P162  u是均值 sigma是标准差
    '''
    def gaussian(self, x, u, sigma):
        return (1.0 / np.sqrt(2 * np.pi)) * (np.exp(-(x - u) ** 2 / (2 * sigma ** 2)))

    '''
    gmm的em迭代算法 公式9.28 公式9.30 9.31 9.32 p165
    alphak是k个高斯分布的概率
    u 是数据均值
    sigma 是标准差
    '''
    def gmm_em(self, alphak, u, sigma):
        n = len(alphak)                          #n个高斯分布密度函数
        dataNum = self.dataArray.size       #训练数目
        gamaArray = np.zeros((n, dataNum))

        for s in range(self.maxSteps):
            for i in range(n):              #计算分模型对观测数据的响应度
                #计算E步：根据当前模型参数，计算分模型i对观测数据的响应度
                for j in range(dataNum):
                    sumAlphaK = sum(alphak[k] * self.gaussian(self.dataArray[j], u[k], sigma[k]) for k in range(n))
                    gamaArray[i][j] = alphak[i] * self.gaussian(self.dataArray[j], u[i], sigma[i]) / float(sumAlphaK)
            #M步：计算新一轮迭代的模型参数
            #更新u
            for i in range(n):
                u[i] = np.sum(gamaArray[i] * self.dataArray) / np.sum(gamaArray[i])
            #更新sigma
            for i in range(n):
                sigma[i] = np.sqrt(np.sum(gamaArray[i] * (self.dataArray - u[i]) ** 2) / np.sum(gamaArray[i]))
            #更新系数alphaK
            for i in range(n):
                alphak[i] = np.sum(gamaArray[i]) / dataNum

        return [alphak, u, sigma]

'''
生成测试数据
'''
def generateData(alphaK, u, sigma, dataNum):
    dataArray = np.zeros(dataNum, dtype=np.float32)

    K = len(alphaK)
    for i in range(dataNum):
        rand = np.random.random()
        Sum = 0
        index = 0
        while index < K:
            Sum += alphaK[index]
            if rand < Sum:
                dataArray[i] = np.random.normal(u[index], sigma[index])
                break
            else:
                index += 1
    return dataArray


if __name__ == '__main__':
    alphaK = [0.3, 0.4, 0.3]
    u = [2, 4, 3]
    sigma = [1, 1, 4]

    dataNum = 5000
    dataArray = generateData(alphaK, u, sigma, dataNum)

    originAlphaK = [0.3, 0.3, 0.4]
    originU = [1, 2, 2]
    originSigma = [1, 1, 1]

    em = GMM(dataArray)
    alphaK1, u1, sigma1 = em.gmm_em(originAlphaK, originU, originSigma)
    print 'real par alphaK ', alphaK
    print 'u ', u
    print 'sigm2 ', sigma
    print 'guess par alphaK', alphaK1
    print 'u ', u1
    print 'sigm2 ', sigma1