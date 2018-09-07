# -*- coding: utf-8 -*-
import numpy as np

class ThreeCoins(object):
    def __init__(self, obs, threshold, maxIters):
        self.obs = obs
        self.threshold = threshold
        self.maxIters = maxIters

    '''
    一次迭代计算
    '''
    def singleEM(self, pi, p, q):
        #计算E步  公式9.5 P156
        uValues = []
        for i in range(len(self.obs)):
            u = pi * (p ** self.obs[i]) * ((1 - p) ** (1 - self.obs[i]))
            u = u / (pi * (p ** self.obs[i]) * ((1 - p) ** (1 - self.obs[i])) + ((1-pi) * (q ** self.obs[i]) * ((1-q) ** (1-self.obs[i]))))
            uValues.append(u)
        #M步 公式9.6 P156
        newPI = 0
        for i in range(len(self.obs)):
            newPI += uValues[i]
        newPI = newPI / len(self.obs)
        #计算p值 公式9.7 P156
        p1 = 0
        p2 = newPI * len(self.obs)
        for i in range(len(self.obs)):
            p1 += uValues[i] * self.obs[i]
        newP = p1 / p2
        #计算q值 公式9.8 P156
        q1 = 0
        q2 = 0
        for i in range(len(self.obs)):
            q1 += (1 - uValues[i]) * self.obs[i]
            q2 += (1 - uValues[i])
        newQ = q1 / q2
        return newPI, newP, newQ

    '''
    EM算法
    '''
    def EM(self, pi, p, q):
        for i in range(self.maxIters):
            newPi, newP, newQ = self.singleEM(pi, p, q)
            delta1 = np.fabs(newPi - pi)
            delta2 = np.fabs(newP - p)
            delta3 = np.fabs(newQ - q)
            pi = newPi
            p = newP
            q = newQ
            if delta1 < self.threshold and delta2 < self.threshold and delta3 < self.threshold:
                break

        print pi, p, q


if __name__ == '__main__':
    obs = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    tc = ThreeCoins(obs, 1e-6, 1000)
    tc.EM(0.46, 0.55, 0.67)