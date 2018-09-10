# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

'''
隐马尔可夫模型（HMM）中的Baum-Welch算法
'''
class HMM_BW(object):
    def __init__(self, N, M):
        self.A = np.zeros((N, N))           #状态转移矩阵概率
        self.B = np.zeros((N, M))           #观测概率矩阵
        self.Pi = np.zeros(N)               #初始状态概率矩阵

        self.N = N                          #状态树
        self.M = M                          #所有可能观测的集合

    '''
    前向算法：P175，由公式 10.15 10.16 10.17可得
    '''
    def forward(self):
        self.alpha = np.zeros((self.T, self.N))     #前向概率

        for i in range(self.N):                     #公式10.15
            self.alpha[0][i] = self.Pi[i] * self.B[i][self.O[0]]

        for t in range(1, self.T):                  #公式10.16
            for i in range(self.N):
                for j in range(self.N):
                    self.alpha[t][i] += self.alpha[t-1][j] * self.A[j][i]
                self.alpha[t][i] *= self.B[i][self.O[t]]

    '''
    后向算法：P178 由公式 10.19 10.20 10.21可得
    '''
    def backward(self):
        self.beta = np.zeros((self.T, self.N))      #后向概率

        for i in range(self.N):                     #公式10.19
            self.beta[self.T - 1][i] = 1

        for t in range(self.T - 2, -1, -1):          #公式10.20
            for i in range(self.N):
                for j in range(self.N):
                    self.beta[t][i] += self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]

    '''
    计算“在时刻t处于qi的概率” 公式 10.24
    '''
    def calGamma(self, i, t):
        p1 = self.alpha[t][i] * self.beta[t][i]

        p2 = 0
        for j in range(self.N):
            p2 += self.alpha[t][j] * self.beta[t][j]

        return p1 / p2

    '''
    计算“在时刻t处于qi状态且在时刻t+1处于qj状态的概率” 公式10.26
    '''
    def calKSI(self, i, j, t):
        p1 = self.alpha[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]

        p2 = 0
        for i1 in range(self.N):
            for j1 in range(self.N):
                p2 += self.alpha[t][i1] * self.A[i1][j1] * self.B[j1][self.O[t+1]] * self.beta[t+1][j1]

        return p1 / p2

    '''
    初始化，主要是A, B, PI的初始化
    '''
    def init(self):
        for i in range(self.N):
            randomArray =  [random.randint(0, 100) for j in range(self.N)]
            Sum = sum(randomArray)
            for j in range(self.N):
                self.A[i][j] = randomArray[j] / float(Sum)                 #记得归一化
        for i in range(self.N):
            randomArray = [random.randint(0, 100) for j in range(self.M)]
            Sum = sum(randomArray)
            for j in range(self.M):
                self.B[i][j] = randomArray[j] / float(Sum)

        randomArray = [random.randint(0, 100) for i in range(self.N)]
        Sum = sum(randomArray)
        for i in range(self.N):
            self.Pi[i] = randomArray[i] / float(Sum)
    '''
    Baum-welch算法 训练过程(EM算法)
    '''
    def train(self, O, maxIter = 100):
        self.T = len(O)
        self.O = O

        self.init()                                         #A, B, PI的初始化
        for ite in range(maxIter):
            tA = np.zeros((self.N, self.N))
            tB = np.zeros((self.N, self.M))
            tPi = np.zeros(self.N)

            #E 步
            self.forward()                                  #前向算法
            self.backward()                                 #后向算法

            #M 步
            #计算a(ij) 公式10.39 P183
            for i in range(self.N):
                for j in range(self.N):
                    p1, p2 = 0.0, 0.0
                    for t in range(self.T - 1):
                        p1 += self.calKSI(i, j, t)          #公式10.26
                        p2 += self.calGamma(i, t)           #公式10.24
                    tA[i][j] = p1 / p2
            #计算b(jk) 公式 10.40 P184
            for j in range(self.N):
                for k in range(self.M):
                    p1, p2 = 0.0, 0.0
                    for t in range(self.T):
                        if k == self.O[t]:
                            p1 += self.calGamma(j, t)
                        p2 += self.calGamma(j, t)
                    tB[j][k] = p1 / p2
            #计算Pi 公式10.41
            for i in range(self.N):
                tPi[i] = self.calGamma(i, 0)

            self.A = tA
            self.B = tB
            self.Pi = tPi

    '''
    生成数据
    '''
    def general(self, len):
        I = []

        ran = random.randint(0, 1000) / 1000.0
        i = 0
        while self.Pi[i] < ran or self.Pi[i] < 0.0001:
            ran -= self.Pi[i]
            i += 1
        I.append(i)

        for i in range(1, len):
            last = I[-1]
            ran = random.randint(0, 1000) / 1000.0
            j = 0
            while self.A[last][j] < ran or self.A[last][j] < 0.0001:
                ran -= self.A[last][j]
                j += 1
            I.append(j)

        Y = []
        for i in range(len):
            k = 0
            ran = random.randint(0, 1000) / 1000.0
            while self.B[I[i]][k] < ran or self.B[I[i]][k] < 0.0001:
                ran -= self.B[I[i]][k]
                k += 1
            Y.append(k)

        return Y

def triangle(len):
    X = [i for i in range(len)]
    Y = []

    for x in X:
        x = x % 6
        if x <= 3:
            Y.append(x)
        else:
            Y.append(6-x)
    return X, Y

def showData(x, y):
    plt.plot(x, y, 'g')
    plt.show()

    return y

if __name__ == '__main__':
    hmm = HMM_BW(10, 4)
    triX, triY = triangle(20)

    hmm.train(triY)
    y = hmm.general(100)
    print y
    x = [i for i in range(100)]
    showData(x, y)