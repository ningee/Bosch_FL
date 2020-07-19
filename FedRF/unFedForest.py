#!/usr/bin/env python
# coding: utf-8

# In[1]:
'''
Federated-Forest
2019.9.25 -2019.10.6
Greilfang
greilfang@gmail.com
'''
# In[2]:
from mpi4py import MPI

# from sklearn import datasets
# from sklearn.datasets.samples_generator import make_blobs
import random
from collections import Counter
import math
import numpy as np
import copy
import pandas as pd
# from simulation import load_random
import time
from assessment import binaryAssessment, find_p


'''
produce the dataset && get the dataset according to index
'''

# 类似于平滑操作，相邻的两个数进行平均，返回一个一维数组，计算gini时使用
def produce_median(vals):
    vals.sort()
    medians = []
    for i in range(1, len(vals)):
        medians.append((vals[i] + vals[i - 1]) / 2)
    return medians


# 将某行，按照某个数进行大小划分，返回划分后的两个数组，计算gini时使用
def splitIndexes(dataset, col, val, indexes):
    ind_2, ind_1 = [], []
    for i in indexes:
        if dataset['data'][i][col] > val:
            # print('2:',dataset['data'][i][col])
            ind_2.append(i)
        else:
            ind_1.append(i)
    return ind_1, ind_2


# 构造包消息


def packMessage(gini, feature, value, rank):
    return {
        'rank': rank,
        'gini': gini,
        'feature': feature,
        'value': value
    }


# 随机选数据（行），在0...data_num-1中,选择selected_num个数，而selected_num等于随机率×data_num


def notify_selected_data(rand_rate, data_num):
    selected_num = int(rand_rate * data_num)
    index = random.sample(range(data_num), selected_num)
    rest_index = list(set(range(data_num)) - set(index))
    random.shuffle(rest_index)
    return index, rest_index


# 随机选数据（列），选rand_rate * feature_num个特征，然后每个客户端特征数为rand_rate * feature_num/n，随机在所有特征里选取rand_rate * feature_num/n特征


def notify_selected_feature(rand_rate, feature_num):
    selected_num = int(rand_rate * feature_num)
    features = random.sample(range(feature_num), selected_num)
    return features


# 计算gini，然后换算成比例，**表示幂运算


def gini(data, ind):
    stats = Counter(data['target'][ind])
    all_nums = len(ind)
    result = 1
    for amt in stats.values():
        result = result - (amt / all_nums) ** 2
    return result


# 得到熵，计算条件熵中使用


def getEntropy(all_nums, class_nums):
    entropy = 0
    for nums in class_nums:
        if nums != 0:
            entropy = entropy - (nums / all_nums) * math.log((nums / all_nums), 2)
    return entropy


# 获得训练集，测试集，rate为test的占比，1-rate为train占比，直接一刀切，在此代码中，使用的那一步已经被注释掉，并未使用


def load_credits(matrix, rate):
    train_set, test_set = {}, {}
    edge = int(matrix.shape[0] * (1 - rate))
    train_set['data'] = matrix[:edge, 1:]
    train_set['target'] = matrix[:edge, 0]
    test_set['data'] = matrix[edge:, 1:]
    test_set['target'] = matrix[edge:, 0]
    return train_set, test_set


# 按条件获得训练集、测试集，条件有：test_num测试样本数量，good_num,bad_num，训练样本的正负样本数量，matrix样本矩阵


def load_propotion_credits(matrix, good_num, bad_num, test_num):
    train_set, test_set = {}, {}
    rows = matrix.shape[0]
    indexes = [i for i in range(rows)]
    test_index = random.sample(indexes, test_num)  # 随机选取test的一些行，共选test_num行

    test_set['data'] = matrix[test_index, 1:]
    test_set['target'] = matrix[test_index, 0]

    remain_index = list(set(indexes) - set(test_index))  # 所有索引中去掉已经选择的索引，将剩下的保留

    good_index, bad_index = [], []
    for index in remain_index:  # 统计剩下的样本中标签为1和不为1的样本索引
        if abs(matrix[index, 0]-1)<0.00000001:
            good_index.append(index)
        else:
            bad_index.append(index)

    good_sample = random.sample(good_index, good_num)  # 标签为1的选出good_num个
    bad_sample = random.sample(bad_index, bad_num)  # 标签不为1的选出bad_num个
    samples = good_sample + bad_sample
    random.shuffle(samples)  # 混洗

    train_set['data'] = matrix[samples, 1:]
    train_set['target'] = matrix[samples, 0]

    return train_set, test_set


# 例如load_equal_credits(matrix,20,10,0.2):训练样本为（20+10），测试样本为（4+2）


def load_equal_credits(matrix, good_num, bad_num, test_ratio):
    train_set, test_set = {}, {}
    rows = matrix.shape[0]
    indexes = [i for i in range(rows)]
    good_index, bad_index = [], []
    for index in indexes:
        if abs(matrix[index, 0]-1)<0.00000001:
            good_index.append(index)
        else:
            bad_index.append(index)

    good_train = random.sample(good_index, int(good_num * (1 - test_ratio)))
    bad_train = random.sample(bad_index, int(bad_num * (1 - test_ratio)))

    rest_good_sample = list(set(good_index) - set(good_train))
    rest_bad_sample = list(set(bad_index) - set(bad_train))

    good_test = random.sample(rest_good_sample, int(good_num * test_ratio))
    bad_test = random.sample(rest_bad_sample, int(bad_num * test_ratio))

    trains = good_train + bad_train
    tests = good_test + bad_test
    random.shuffle(trains)
    random.shuffle(tests)

    train_set['data'] = matrix[trains, 1:]
    train_set['target'] = matrix[trains, 0]
    test_set['data'] = matrix[tests, 1:]
    test_set['target'] = matrix[tests, 0]

    return train_set, test_set


# 联邦决策树分类器


class FederatedDecisionTreeClassifier:
    def __init__(self, dataset):
        # dataset is the actual dataset a client owns
        self.dataset = dataset
        # classifier structure
        self.structure = None
        # prevent the repetition in split threshold
        self.threshold_map = {}
        self.tests = None
        self.leaves = []

    # 需要修剪

    def need_pruning(self, indexes):
        if len(indexes) < 4:
            return True
        return False

    # 套袋

    def bagging(self, indexes):
        # print('rank： ',self.rank,' ',indexes)
        stats = Counter(self.dataset['target'][indexes])
        if stats[0] >= stats[1]:
            return 0
        else:
            return 1
        # prediction = max(stats,key = stats.get)
        # return prediction

    # 判断feature[i]应该落在左节点还是右节点

    def repeated(self, features, i, val):
        if features[i] in self.threshold_map and val in self.threshold_map[features[i]]:
            return True
        return False

    # 计算最佳gini
    def calculateBestGini(self, indexes, features):
        rows= len(indexes)
        best_gini_gain, split_feature, split_value = float('Inf'), -1, None
        # print('indexes:\n',indexes)
        for i in features:
            # medians = produce_median(list(set(self.dataset['data'][indexes, i])))
            newindex = np.lexsort((indexes,self.dataset['data'][indexes,i]))
            tmpind2 = Counter(self.dataset['target'][indexes])
            tmpind1 = Counter()
            for j in range(rows-1):
                val1 = self.dataset['data'][indexes[newindex[j]],i]
                val2 = self.dataset['data'][indexes[newindex[j+1]],i]
                split_left = self.dataset['target'][indexes[newindex[j]]]
                tmpind1.update([split_left])
                tmpind2.subtract([split_left])
                if abs(val1-val2)<0.00000000001:
                    continue
                p = (j+1)/rows
                result1, result2 = 1,1
                for amt in tmpind1.values():
                    result1 -= (amt / (j+1)) ** 2
                for amt in tmpind2.values():
                    result2 -= (amt / (rows-j-1)) ** 2
                if result1<0:
                    print(tmpind1,j+1)
                if result2<0:
                    print(tmpind2,(rows-j-1))

                gini_gain = p * result1 + (1-p) * result2
                if gini_gain < best_gini_gain :
                    best_gini_gain, split_feature, split_value = gini_gain, i, \
                        (self.dataset['data'][indexes[newindex[j]],i]+self.dataset['data'][indexes[newindex[j+1]],i])/2

            # for val in medians:
            #     ind_1, ind_2 = splitIndexes(self.dataset, i, val, indexes)
            #     p = len(ind_1) / rows
            #     gini_gain = p * gini(self.dataset, ind_1) + (1 - p) * gini(self.dataset, ind_2)
            #     if gini_gain < best_gini_gain :
            #         best_gini_gain, split_feature, split_value = gini_gain, i, val
        return best_gini_gain, split_feature, split_value

    # 构建树

    def buildTree(self, indexes, rest_index, features):
        self.structure = self.FederatedTreeBuild(indexes, rest_index, features)

    # dataset will change through the split of tree，构建联邦树，rank=0表示中心

    def FederatedTreeBuild(self, indexes, rest_index, features):
        # print('\nindex:',len(indexes),len(rest_index))
        node = {'feature': None, 'threshold': None, 'gini': None, 'left': None, 'right': None}
        if len(set(self.dataset['target'][indexes])) == 1:
            # 本次划分中分类全部一致则生成叶子结点
            ind = indexes[0]
            node['class'] = self.dataset['target'][ind]
            return node
        if len(indexes)<3:
            node['class'] = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
            return node

        best_gini_gain, split_feature, split_value = None, None, None
        best_gini_gain, split_feature, split_value = self.calculateBestGini(indexes, features)  # 取得gini指数最小的特征

        # 为防止出现所有feature上数值均相等的情况(实际上概率几乎为0)
        if split_value is None:
            this_class = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
            node['class'] = this_class
            return node
        # end
        # print('split value:',split_value)
        # print('split feature:',self.dataset['data'][indexes,split_feature])
        left_indexes, right_indexes = splitIndexes(self.dataset, split_feature, split_value, indexes)
        # print('left:',len(left_indexes))
        # print('right:',len(right_indexes))
        # pre_pruning start
        rest_left, rest_right = splitIndexes(self.dataset, split_feature, split_value, rest_index)
        if len(rest_left) != 0 and len(rest_right) != 0:
            this_class = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
            left_class = Counter(self.dataset['target'][left_indexes]).most_common(1)[0][0]
            right_class = Counter(self.dataset['target'][right_indexes]).most_common(1)[0][0]
            prune_acc = Counter(self.dataset['target'][rest_index])[this_class] / len(rest_index)
            grow_acc = (Counter(self.dataset['target'][rest_left])[left_class] +
                        Counter(self.dataset['target'][rest_right])[right_class]) / len(rest_index)
            if prune_acc > grow_acc or prune_acc>0.9:
                node['class'] = this_class
                return node
        # pre_pruning end

        node['feature'], node['threshold'], node['gini'] = split_feature, split_value, best_gini_gain

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            this_class = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
            node = {'feature': None, 'threshold': None, 'gini': None, 'left': None, 'right': None, 'class': this_class}
            return node
        print('left:', len(left_indexes))
        node['left'] = self.FederatedTreeBuild(left_indexes, rest_left, features)
        print('right:', len(right_indexes))
        node['right'] = self.FederatedTreeBuild(right_indexes, rest_right, features)
        return node

    # 封装预测函数

    def predict(self, dataset):
        self.tests = dataset
        indexes = [i for i in range(dataset['data'].shape[0])]
        self.FederatedTreePredict(indexes, self.structure)

    # 封装联邦树预测函数

    def FederatedTreePredict(self, indexes, node):
        if 'class' in node:
            self.leaves.append({'indexes': indexes, 'prediction': node['class']})
        # elif node['feature'] is None:
        #     self.FederatedTreePredict(copy.deepcopy(indexes), node['left'])
        #     self.FederatedTreePredict(copy.deepcopy(indexes), node['right'])
        elif node['feature'] is not None:
            left_ind, right_ind = splitIndexes(self.tests, node['feature'], node['threshold'], indexes)
            self.FederatedTreePredict(left_ind, node['left'])
            self.FederatedTreePredict(right_ind, node['right'])


# 联邦森林分类器


class FederatedForestClassifier:
    def __init__(self, n_tree, rate, frate):
        self.forest = []
        self.tree_num = n_tree
        self.rate = rate
        self.frate = frate
        self.voters = None
        self.dataset = None
        self.tests = None

    def fit(self, dataset):
        self.dataset = dataset
        data_num = self.dataset['data'].shape[0]
        feature_num = self.dataset['data'].shape[1]
        # self.voters = np.zeros((data_num,self.tree_num))

        for i in range(self.tree_num):
            print('Tree ', i)
            print('--------------------------------------------------------------------------')
            selected_index = []
            selected_index, rest_index = notify_selected_data(self.rate, data_num)  # 根据随机度从整体数据选出一定的行
            selected_feature = notify_selected_feature(self.frate, feature_num)

            fdtc = FederatedDecisionTreeClassifier(copy.deepcopy(self.dataset))

            fdtc.buildTree(selected_index, rest_index, selected_feature)
            self.forest.append(fdtc)

    def predict(self, dataset):
        self.tests = dataset
        data_num = self.tests['data'].shape[0]
        self.voters = np.zeros((data_num, self.tree_num))
        for i in range(self.tree_num):
            classify_result = []
            self.forest[i].predict(dataset)
            classify_result = self.forest[i].leaves
            self.generateTarget(i, classify_result)

        self.bagging()

    def generateTarget(self, tree, real_leafs):
        for real_leaf in real_leafs:
            self.voters[real_leaf['indexes'], tree] = real_leaf['prediction']

    def bagging(self):
        test_num = self.tests['data'].shape[0]
        self.prediction = [-1 for x in range(test_num)]
        self.predict_prob = [-1 for x in range(test_num)]
        for i in range(test_num):
            self.prediction[i] = max(self.voters[i, :], key=list(self.voters[i]).count)
            tmpsum = 0
            for j in self.voters[i, :]:
                if abs(j-1.0)<0.00000001:
                    tmpsum += 1
            self.predict_prob[i] = tmpsum / len(self.voters[i, :])
        # print('final bagging\n',self.prediction)

    def getAccuracy(self, target):
        re, pred = find_p(self.predict_prob, target)
        # # binaryAssessment(self.prediction,target)
        # all_predict, true_predict = 0, 0
        # class_0, class_1 = 0, 0
        # acc_0, acc_1 = 0, 0
        # assert (len(self.prediction) == len(target))
        # for i in range(len(self.prediction)):
        #     if target[i] == 0:
        #         class_0 = class_0 + 1
        #     elif target[i] == 1:
        #         class_1 = class_1 + 1
        #
        #     if self.prediction[i] == target[i]:
        #         if self.prediction[i] == 0:
        #             acc_0 = acc_0 + 1
        #         elif self.prediction[i] == 1:
        #             acc_1 = acc_1 + 1
        #         true_predict = true_predict + 1
        #     all_predict = all_predict + 1
        # print('accuracy:', true_predict / all_predict)
        # print('0 accuracy:', acc_0 / class_0)
        # print('1 accuracy:', acc_1 / class_1)
        return re, pred

hyperparams = {
    'client_num': 1,
    'tree_num': 10,  # 树数量
    'rand_data_rate': 0.7,  # 随机度
    'rand_feature_rate': 0.9  # 特征随机度
}

result_all = []
pred_all = []

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
train_set, test_set = {}, {}
train_set['data'] = train.iloc[:, 1:].values
train_set['target'] = train.iloc[:, 0].values
test_set['data'] = test.iloc[:, 1:].values
test_set['target'] = test.iloc[:, 0].values

for turn in range(1):

    timw_start, time_end = None, None
    time_start = time.time()
    print("start time:", time_start)

    ffc = FederatedForestClassifier(
    n_tree=hyperparams['tree_num'],
    rate=hyperparams['rand_data_rate'],
    frate=hyperparams['rand_feature_rate']
    )
    ffc.fit(copy.deepcopy(train_set))
    ffc.predict(test_set)

    time_end = time.time()
    result, pred = ffc.getAccuracy(test_set['target'])
    result_all.append(result)
    pred_all.append(pred)
    print('total time:', time_end - time_start)

print('result:')
print(result_all)
print('best:')
final_re = [0,0,0,0,0]
final_pred = None
for i in range(len(result_all)):
    if final_re[3] < result_all[i][3]:
        final_re = result_all[i]
        final_pred = pred_all[i]
print(final_re)
print(final_pred)
gt = pd.DataFrame(test_set['target'],index=test.index,columns=['GT'])
gt['unfed'] = final_pred
gt.sort_index(inplace=True)
print(gt)
gt.to_csv('./markov/FedForest_result_unfed.csv')
