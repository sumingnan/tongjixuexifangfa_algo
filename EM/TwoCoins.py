# -*- coding: utf-8 -*-
import numpy as np

class TwoCoins(object):
    def __init__(self, obs, threshold, maxIter):
        self.obs = obs
        self.threshold = threshold
        self.maxIter = maxIter

    '''
    计算PMF公式
    '''
    def pmf(self, k, n, p):
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n-k)) * (p ** k)*((1-p) ** (n-k))

    '''
    迭代一次EM算法
    '''
    def calSingleEM(self, thetaA, thetaB):
        counts = {'A':{'H':0, 'T':0}, 'B':{'H':0, 'T':0}}               #H代表正面，T代表反面

        #E步计算
        for ob in self.obs:
            count = len(ob)
            head = ob.sum()
            tail = count - head
            pmfA = self.pmf(head, count, thetaA)
            pmfB = self.pmf(head, count, thetaB)
            weightA = pmfA / (pmfA + pmfB)
            weightB = pmfB / (pmfA + pmfB)
            counts['A']['H'] += weightA * head
            counts['A']['T'] += weightA * tail
            counts['B']['H'] += weightB * head
            counts['B']['T'] += weightB * tail

        #M步更新
        newThetaA = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
        newThetaB = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])

        return newThetaA, newThetaB

    def trainEM(self, oriThetaA, oriThetaB):
        for i in range(self.maxIter):
            #一次迭代
            newThetaA, newThetaB = self.calSingleEM(oriThetaA, oriThetaB)
            delta = np.abs(newThetaA - oriThetaA)
            oriThetaA = newThetaA
            oriThetaB = newThetaB
            if delta < self.threshold:
                break
        print oriThetaA, oriThetaB


if __name__ == '__main__':
    #观察结果
    obs = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
    tc = TwoCoins(obs, 1e-6, 10000)
    tc.trainEM(0.6, 0.5)
