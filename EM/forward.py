# -*- coding: utf-8 -*-

import numpy as np

class Forward(object):
    def __init__(self, A, B, Pi):
        self.A = A                          #观测概率分布
        self.B = B                          #状态转移概率分布
        self.Pi = Pi                        #初始概率分布

    '''
    训练计算观测概率
    o是观测序列 红-0 白-1
    '''
    def train(self, o):
        #1. 计算初值
        alpha = np.zeros((len(o), len(self.A[0])))
        for i in range(len(self.A[0])):
            alpha[0][i] = self.Pi[i] * self.B[i][o[0]]
        #2. 递推计算
        for i in range(1, len(o)):
            for j in range(len(self.A[0])):
                for k in range(len(self.A[0])):
                    alpha[i][j] += alpha[i-1][k] * self.A[k][j]
                alpha[i][j] *= self.B[j][o[i]]
        #3. 计算观测概率
        p = 0
        for i in range(len(self.A[0])):
            p += alpha[len(o)-1][i]

        return p

if __name__ == '__main__':
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])

    o = np.array([0, 1, 0])
    f = Forward(A, B, pi)
    p = f.train(o)
    print p
