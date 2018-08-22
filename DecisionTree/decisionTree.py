# -*- coding: utf-8 -*-
from math import log
import operator
import pickle
import copy
import numpy as np

#选取信息增益比作为选取准则，可以校正ID3算法的问题
'''
计算经验熵，此对数以2为底，因此熵的单位为比特，如果以e为底，则单位为纳特
'''
def calcShannonEnt(dataSet):
    numTests = len(dataSet)
    typeCounts = {}
    for data in dataSet:
        currentType = data[-1]
        if currentType not in typeCounts.keys():
            typeCounts[currentType] = 0
        typeCounts[currentType] += 1

    entropy = 0.0
    for key in typeCounts:
        property = float(typeCounts[key]) / numTests
        entropy -= property * log(property, 2)

    return entropy

'''
选取某个特征的特征值子数列
'''
def getSubDataSet(dataSet, featureIndex, featureValue):
    subDataSet = []
    for feature in dataSet:
        if feature[featureIndex] == featureValue:
            subFeature = feature[:featureIndex]
            subFeature.extend(feature[featureIndex+1:])
            subDataSet.append(subFeature)
    return subDataSet

'''
根据特征值划分数据集，返回的是不等于该特征值的子集
'''
def getOtherDataSet(dataSet, featureIndex, featureValue):
    subDataSet = []                                 #返回的数据子集
    for data in dataSet:
        if data[featureIndex] != featureValue:
            rowFeatureVec = data[:featureIndex]
            rowFeatureVec.extend(data[featureIndex+1:])
            subDataSet.append(rowFeatureVec)
    return subDataSet

'''
计算基尼指数
'''
def calcGini(dataSet):
    numOfData = len(dataSet)
    typeCounts = {}
    for data in dataSet:
        currentType = data[-1]                      #最后一列为类别
        if currentType not in typeCounts.keys():
            typeCounts[currentType] = 0
        typeCounts[currentType] += 1

    giniValue = 1.0
    for type in typeCounts:
        prop = float(typeCounts[type]) / numOfData
        giniValue -= prop * prop
    return giniValue

'''
选择信息增益最大的特征作为构建子节点，返回该特征索引
'''
def chooseBestFeatureByID3(dataSet):
    numFeature = len(dataSet[0]) - 1            #最后一列为列标签
    baseEntropy = calcShannonEnt(dataSet)       #计算经验熵
    bestInfoGain = 0.0
    bestFeature = -1
    #print "base = ", baseEntropy
    for i in range(numFeature):                 #遍历每个特征，计算其经验条件熵
        featureValues = [example[i] for example in dataSet]  #获取该特征的所有特征值
        uniqueValues = set(featureValues)                   #获取该特征值的种类（过滤掉重复值）
        featureEntropy = 0.0                                #该特征的经验熵
        for value in uniqueValues:
            subDataSet = getSubDataSet(dataSet, i, value)   #获取该特征值子数列
            #计算经验条件熵
            property = len(subDataSet) / float(len(dataSet))
            featureEntropy += property * calcShannonEnt(subDataSet)
        inforGain = baseEntropy - featureEntropy
       # print "i = ", i, inforGain
        if inforGain > bestInfoGain:
            bestInfoGain = inforGain
            bestFeature = i
    return bestFeature, bestInfoGain

'''
选择信息增益比最大的特征作为构建子节点，返回该特征索引
'''
def chooseBestFeatureByC45(dataSet):
    numFeature = len(dataSet[0]) - 1            #最后一列为列标签
    baseEntropy = calcShannonEnt(dataSet)       #计算经验熵
    bestInfoGainRatio = 0.0
    bestFeature = -1
    #print "base = ", baseEntropy
    for i in range(numFeature):                 #遍历每个特征，计算其经验条件熵
        featureValues = [example[i] for example in dataSet]  #获取该特征的所有特征值
        uniqueValues = set(featureValues)                   #获取该特征值的种类（过滤掉重复值）
        featureEntropy = 0.0                                #该特征的经验熵
        entropyA = 0.0
        for value in uniqueValues:
            subDataSet = getSubDataSet(dataSet, i, value)   #获取该特征值子数列
            #计算经验条件熵
            property = len(subDataSet) / float(len(dataSet))
            featureEntropy += property * calcShannonEnt(subDataSet)
            entropyA -= property * log(property, 2)         #计算特征A的熵
        inforGainRatio = (baseEntropy - featureEntropy) / entropyA  #计算信息增益比
       # print "i = ", i, inforGainRatio
        if inforGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = inforGainRatio
            bestFeature = i
    return bestFeature, bestInfoGainRatio

'''
根据最优特征的最优切分点二值化数据集
'''
def binaryZationDataSet(dataSet, bestFeature, bestFeatureValue):
    featureValueList = [data[bestFeature] for data in dataSet]
    uniquedValues = set(featureValueList)

    if len(uniquedValues) >= 2:
        for i in range(len(dataSet)):
            if dataSet[i][bestFeature] == bestFeatureValue:
                pass
            else:
                dataSet[i][bestFeature] = 'other'

'''
选择基尼指数最小的特征及对应的切分点作为最优特征与最优切分点
'''
def chooseBestFeatureByCART(dataSet):
    numFeature = len(dataSet[0]) - 1  # 最后一列为列标签
    bestGiniCut = 100000.0
    bestGiniCutValue = []
    bestFeature = -1
    for i in range(numFeature):
        featureValueLists = [data[i] for data in dataSet]
        uniqueFeatureFeatureValues = set(featureValueLists)                     #针对离散的变量
        for value in uniqueFeatureFeatureValues:                                #计算该特征下，每个特征值的基尼指数
            subDataSet = getSubDataSet(dataSet, i, value)                       #获取该特征值下的子数据
            prob = len(subDataSet) / float(len(dataSet))
            giniValue1 = prob * calcGini(subDataSet)                            #计算等于该特征值下的基尼指数

            otherSubDataSet = getOtherDataSet(dataSet, i, value)                #获取不等于特征值的子数据
            prob = len(otherSubDataSet) / float(len(dataSet))
            giniValue2 = prob * calcGini(otherSubDataSet)                       #获取不等于该特征值下的基尼指数部分
            giniValue = giniValue1 + giniValue2                                 #计算该特征值的基尼指数
            if giniValue < bestGiniCut:
                bestGiniCut = giniValue
                bestGiniCutValue = value
                bestFeature = i
    #print bestFeature, bestGiniCutValue, bestGiniCut
    binaryZationDataSet(dataSet, bestFeature, bestGiniCutValue)
    return bestFeature, 1.0

'''
返回该数据集中类别最多的类返回
'''
def mostClassType(classList):
    classCount = {}
    for class_i in classList:
        if class_i not in classCount.keys():
            classCount[class_i] = 0
        classCount[class_i] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
创建决策树
'''
def createDecisionTree(dataSet, labels, alpha, chooseBestFeatureToSplitFunc = chooseBestFeatureByID3):
    typeList = [example[-1] for example in dataSet]         #获取列表
    if typeList.count(typeList[0]) == len(typeList):        #如果只有一个类别，则直接返回该类别
        return typeList[0]

    if len(dataSet[0]) == 1:                                #如果没有特征值
        return mostClassType(typeList)

    bestFeature, bestInfoGain = chooseBestFeatureToSplitFunc(dataSet)  #选取信息增益最大的特征
    if bestInfoGain < alpha:                                #如果特征的信息增益比小于阈值西格玛alpha，则返回实例树中最大类别
        return mostClassType(typeList)
    bestFeatureLabel = labels[bestFeature]                  #得到该特征
    decTree = {bestFeatureLabel:{}}                         #定义树结构
    #print bestFeatureLabel
    bestFeatureValues = [example[bestFeature] for example in dataSet]   #获取所有的特征值
    uniqueFeatureValues = set(bestFeatureValues)                        #获取特征值，过滤掉重复的特征
    for value in uniqueFeatureValues:
        subDataSet = getSubDataSet(dataSet, bestFeature, value)         #获取每个特征值的子数据集
        subLabels = labels[:]
        decTree[bestFeatureLabel][value] = createDecisionTree(subDataSet, subLabels, alpha, chooseBestFeatureToSplitFunc)    #递归创建树
    return decTree

'''
查找当前标签在决策树中的节点
'''
def mapFeatureToLabelIndex(tree, labels):
    for key in tree.keys():
        for i in range(len(labels)):
            if key == labels[i]:
                return key, i
'''
决策树预测函数
'''
def predict(testData, myTree, labels):
    feature_label, feature_index = mapFeatureToLabelIndex(myTree, labels)       #获取决策树节点的下标
    subTree = myTree[feature_label][testData[feature_index]]
    if ~isinstance(subTree, dict):                                              #判断是否是叶节点
        return subTree                                                             #如果是叶节点直接返回
    else:
        return predict(testData, subTree, labels)                               #如果不是叶节点，递归调用

'''
cart决策树预测函数
'''
def cartPredict(testData, myTree, labels):
    feature_label, feature_index = mapFeatureToLabelIndex(myTree, labels)
    if testData[feature_index] not in myTree[feature_label]:
        testData[feature_index] = 'other'
    tree = myTree[feature_label][testData[feature_index]]
    if not isinstance(tree, dict):
        return tree
    else:
        return cartPredict(testData, tree, labels)

'''
决策树准确率判断
'''
def calPredict(dataSet, predictSet):
    length = len(dataSet)
    count = 0
    for i in range(length):
        if dataSet[i][-1] == predictSet[i]:                                      #预测对了
            count += 1
    return count / float(length) * 100.0

'''
创建训练树集
'''
def createDataSet(fileName):
    '''
    [
    age: 1->青年 2->中年  3->老年
    work: 1->是  0->否
    house：1->是  0->否
    credit：1->一般 2->好    3->非常好
    ]

    dataSet = [[1, 0, 0, 1, "no"], [1, 0, 0, 2, "no"], [1, 1, 0, 2, "yes"], [1, 1, 1, 1, "yes"], [1, 0, 0, 1, "no"],
               [2, 0, 0, 1, "no"], [2, 0, 0, 2, "no"], [2, 1, 1, 2, "yes"], [2, 0, 1, 3, "yes"], [2, 0, 1, 3, "yes"],
               [3, 0, 1, 3, "yes"], [3, 0, 1, 2, "yes"], [3, 1, 0, 2, "yes"], [3, 1, 0, 3, "yes"], [3, 0, 0, 1, "no"]]
    labels = ["age", "work", "house", "credit"]

    return dataSet, labels
    '''
    dataSet = []
    labels = []
    with open(fileName) as ifile:
        for line in ifile:
            tokens = line.strip().split(',')
            dataSet.append(tokens)
    labels = ['buying','maint','doors','persons','lug_boot','safety']
    return dataSet, labels

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="round4", color='#3366FF')  #定义判断结点形态
leafNode = dict(boxstyle="circle", color='#FF6633')  #定义叶结点形态
arrow_args = dict(arrowstyle="<-", color='g')  #定义箭头

#绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


#计算叶结点数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


#计算树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


#在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  #在父子结点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #绘制带箭头的注释
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == '__main__':
    dataSet, labels = createDataSet("car.data")
    predict_labels = copy.copy(labels)
    np.random.shuffle(dataSet)

    m = len(dataSet)
    rate = 0.7
    training_len = int(m * rate)

    trainingSet, testSet = dataSet[0:training_len], dataSet[training_len+1:-1]
    decTree = createDecisionTree(trainingSet, labels, 0.1, chooseBestFeatureByCART)

    from pylab import *

    createPlot(decTree)

    predict_results = []
    for data in testSet:
        result = cartPredict(data[0:-1], decTree, predict_labels)
        predict_results.append(result)

    print "decision Tree predict precision: %.2f "%calPredict(testSet, predict_results), "%"
