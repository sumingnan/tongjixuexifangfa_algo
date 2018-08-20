# -*- coding: utf-8 -*-

import sys
from math import sqrt
from collections import namedtuple
from time import clock
from random import random

reload(sys)
sys.setdefaultencoding('utf8')

# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited")

#kd-tree节点的数据结构
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt   #k维向量节点（样本点）
        self.split = split #整数（进行分割纬度的序号
        self.left = left #该节点分割左超平面的kd-tree节点
        self.right = right #该节点分割右超平面的kd-tree节点

#递归构建KD树
class KdTree(object):
    def __init__(self, data):
        k = len(data[0])                #数据维度

        def CreateNode(split, data_set):  #按第split维划分数据集set创建kd-node
            if not data_set:    #如果数据集为空
                return None
            #key参数的值为一函数，此函数只有一个参数并且返回一个值用来比较
            #operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的对象中的序号
            #data_set
            data_set.sort(key=lambda x : x[split])
            split_pos = len(data_set) // 2  #//为python的整数除法
            median = data_set[split_pos]    #中位数
            split_next = (split + 1) % k  # cycle coordinate

            #递归的创建kd树
            return KdNode(median, split,
                          CreateNode(split_next, data_set[:split_pos]), #创建左子树
                          CreateNode(split_next, data_set[split_pos+1:]))   #创建右子树

        self.root = CreateNode(0, data)   # 从第0维分量开始构建kd树，返回根节点

#kd-tree的前序排列
def preoder(root):
    print root.dom_elt

    if root.left:   #节点不为空
        preoder(root.left)

    if root.right:
        preoder(root.right)

def find_nearest(tree, point):
    k = len(point)  #数据维度
    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0) #python 用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split                   #进行分割的数据维度
        pivot = kd_node.dom_elt             #进行分割的轴

        if target[s] <= pivot[s]:           #如果目标点第s维小于分割轴的对应值（目标离左子树更近）
            nearer_node = kd_node.left      #下一个访问节点为左子树根节点
            further_node = kd_node.right    #同时记录下右子树
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)   #进行遍历找到含有目标点的区域

        nearest = temp1.nearest_point                    #以此叶节点作为"当前最近点"
        dist = temp1.nearest_dist                       #更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist                                 #最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])               #第s维上目标点与分割面的距离
        if max_dist < temp_dist:
            return result(nearest, dist, nodes_visited)     #不相交则可以直接返回，不用继续判断

        #计算目标点与分割点的欧式距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:            #如果更近
            nearest = pivot             #更新更近点
            dist = temp_dist             #更新最近距离
            max_dist = dist             #更新超球体半径

        #检查另一个子节点对应的区域是否右更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:           #如果另一个节点存在更近距离
            nearest = temp2.nearest_point       #更新最近点
            dist = temp2.nearest_dist           #更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf")) #从根节点开始递归

if __name__ == "__main__":
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]  # samples

    kd = KdTree(data)
    ret = find_nearest(kd, [3, 4.5])
    print ret

