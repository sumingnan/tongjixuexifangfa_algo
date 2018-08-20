# -*- coding: utf-8 -*-
import numpy as np
#朴素贝叶斯估计算法
class NaviteBayes(object):

    def findNumIndex(self, labels, value):
        for i in range(len(labels)):
            if labels[i] == value:
                return i
        return 0

    def train(self, dataSet, classLabels, featureNum, featureVector, labelVector):
        trainNum = len(dataSet)
        class_num = len(labelVector)

        #找出特征值取值最多的数目
        maxFeatureValueCount = len(featureVector[0])
        for i in range(featureNum):
            if len(featureVector[i]) > maxFeatureValueCount:
                maxFeatureValueCount = featureVector[i]

        prior_probabilityCount = np.zeros(class_num)                                                 #先验概率统计
        conditional_probabilityCount = np.zeros((class_num, featureNum, maxFeatureValueCount))      #条件概率统计
        #统计先验概率数据
        for i in range(trainNum):
            rLabel = self.findNumIndex(labelVector, classLabels[i])
            prior_probabilityCount[rLabel] += 1

            for j in range(featureNum):
                feature = dataSet[i]
                rValue = self.findNumIndex(featureVector[j], feature[j])
                conditional_probabilityCount[rLabel][j][rValue] += 1
        #  1. 计算先验概率
        self.prior_probability = np.zeros(class_num)                                             # 先验概率
        self.conditional_probability = np.zeros((class_num, featureNum, len(featureVector[0])))  # 条件概率
        for i in range(class_num):
            self.prior_probability[i] = prior_probabilityCount[i] / trainNum

        # 2. 计算条件概率
        for i in range(class_num):
            for j in range(featureNum):
                for k in range(len(featureVector[j])):
                    self.conditional_probability[i][j][k] = conditional_probabilityCount[i][j][k] / prior_probabilityCount[i]
        '''
        for i in range(class_num):
            print "label ", labelVector[i]
            for j in range(featureNum):
                print "feature ", j
                for k in range(len(featureVector[0])):
                    print conditional_probabilityCount[i][j][k]
        '''
    def predict(self, featureVector, labels, test):
        # 3. 计算各个类的概率
        probability = np.zeros(len(self.prior_probability))
        for i in range(len(self.prior_probability)):
            probability[i] = self.prior_probability[i]
            for j in range(len(test)):
                rValue = self.findNumIndex(featureVector[j], test[j])
                probability[i] = probability[i] * self.conditional_probability[i][j][rValue]
        # 4. 确定实例分类点
        maxPrior = 0
        maxIndex = -1
        for i in range(len(probability)):
            if probability[i] > maxPrior:
                maxPrior = probability[i]
                maxIndex = i
        return labels[maxIndex]
        '''
        for i in range(len(self.prior_probability)):
            print probability[i]
        '''

if __name__ == "__main__":
    data = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]  # data
    labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    #训练
    bayes = NaviteBayes()
    bayes.train(data, labels, 2, [[1, 2, 3], ['S', 'M', 'L']], [-1, 1])
    #预测
    result = bayes.predict([[1, 2, 3], ['S', 'M', 'L']], [-1, 1], [2, 'S'])
    print "y = ", result