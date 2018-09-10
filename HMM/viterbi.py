# -*- coding: utf-8 -*-

import numpy as np

class Viterbi(object):
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

        self.N = len(A[0])          #状态数
        self.M = len(B[0])          #观测集合数目

    '''
    训练
    '''
    def train(self, o):
        T = len(o)
        sek = np.zeros((self.N, T))
        index = np.zeros((self.N, T), dtype=int)
        #1. 初始化
        for i in range(self.N):
            sek[i][0] = self.Pi[i] * self.B[i][o[0]]
            index[i][0] = 0
        #2. 递推
        for t in range(1, T):
            for i in range(self.N):
                sekMax = -np.inf
                sekMaxIndex = -1
                for j in range(self.N):
                    p = sek[j][t-1] * self.A[j][i]
                    if p > sekMax:
                        sekMax = p
                        sekMaxIndex = j
                sek[i][t] = sekMax * self.B[i][o[t]]
                index[i][t] = sekMaxIndex + 1
        #3. 终止
        maxP = -np.inf
        maxPindexs = np.zeros(T, dtype=int)
        for i in range(self.N):
            if sek[i][T-1] > maxP:
                maxP = sek[i][T-1]
                maxPindexs[T-1] = int(i+1)
        #4. 最优路径回溯
        for t in range(T-2, -1, -1):
            pp = maxPindexs[t+1] - 1
            maxPindexs[t] = index[2][t+1]

        return maxPindexs

if __name__ == '__main__':
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    Pi = np.array([0.2, 0.4, 0.4])

    o = np.array([0, 1, 0, 1])

    v = Viterbi(A, B, Pi)
    path = v.train(o)
    print path