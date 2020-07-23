#!/usr/bin/env python
# coding: utf-8

# In[1]:
'''
Federated-Forest
2019.9.25 -2019.10.6
Greilfang
greilfang@gmail.com

2020.05.01 - 2020.05.30
Improved by Guanghao Li
liguanghao@buaa.edu.cn
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
from assessment import binaryAssessment,find_p
# In[3]:


'''
produce the dataset && get the dataset according to index
'''
# 处理数据，hyperparams是超参数，在497行定义，最终返回处理好的包含多个客户端的数据集，以及各个客户端的特征列表
def simulated_split_dataset(digit):
    datasets = [{} for c in range(0, hyperparams['client_num']+1)]
    all_features=[[] for c in range(0, hyperparams['client_num']+1)]
    datasets[0] = digit
    for client in range(1, hyperparams['client_num']+1):  # 对datasets进行填充，先填充Label
        datasets[client]['target'] = digit['target']
        datasets[client]['data'] = None
    if(hyperparams['client_num']!=0):
        # for i in range(digit['data'].shape[1]):  # 对列进行客户端划分，等间距抽样
        #     client = i % hyperparams['client_num'] + 1
        #     # print('check none:',datasets[client]['data'])
        #     if datasets[client]['data'] is None:
        #         datasets[client]['data'] = digit['data'][:, i]
        #     else:
        #         datasets[client]['data'] = np.column_stack([datasets[client]['data'], digit['data'][:, i]])
        #     all_features[client].append(i)
        col_num_mean = int(digit['data'].shape[1]/hyperparams['client_num'])
        for i in range(hyperparams['client_num']):
            datasets[i+1]['data'] = digit['data'][:,i*col_num_mean:(i+1)*col_num_mean]
            for j in range(i*col_num_mean,(i+1)*col_num_mean):
                all_features[i+1].append(j)
    else:
        print("No client! No Split data!")
    return datasets, all_features
# 类似于平滑操作，相邻的两个数进行平均，返回一个一维数组，计算gini时使用


def produce_median(vals):
    vals.sort()
    medians=[]
    for i in range(1,len(vals)):
        medians.append((vals[i]+vals[i-1])/2)
    return medians
# 将某行，按照某个数进行大小划分，返回划分后的两个数组，计算gini时使用


def splitIndexes(dataset, col, val, indexes):
    ind_2, ind_1 = [], []
    for i in indexes:
        if dataset['data'][i][col] > val:
            ind_2.append(i)
        else:
            ind_1.append(i)
    return ind_1, ind_2
# 构造包消息


def packMessage(gini, feature, value, rank, need_prune):
    return{
        'rank': rank,
        'gini': gini,
        'feature': feature,
        'value': value,
        'need_prune': need_prune
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
    selected_features = [[]]
    for i in range(1, hyperparams['client_num']+1):
        selected_num = int(rand_rate * feature_num / hyperparams['client_num'] )
        features = random.sample(all_features[i], selected_num)
        selected_features.append(features)
        selected_features[0].extend(features)
    
    return selected_features
# 计算gini，然后换算成比例，**表示幂运算


def gini(data, ind):
    stats = Counter(data['target'][ind])
    all_nums = len(ind)
    result = 1
    for amt in stats.values():
        result = result-(amt/all_nums)**2
    return result
# 得到熵，计算条件熵中使用


def getEntropy(all_nums, class_nums):
    entropy = 0
    for nums in class_nums:
        if nums != 0:
            entropy = entropy - (nums/all_nums) * math.log((nums/all_nums), 2)
    return entropy
# 得到条件熵，计算最佳信息增益时会用到


def getConditionEntropy(dataset, col, all_nums):
    fc_record = {}
    for i in range(len(dataset['target'])):
        val, cls = dataset['data'][i][col], dataset['target'][i]
        if dataset['data'][i][col] in fc_record:
            fc_record[val][cls] = fc_record[val][cls]+1
        else:
            fc_record[val] = [0 for c in range(datparams['class_num'])]
            fc_record[val][cls] = 1
    condition_entropy = 0
    for val, cls in zip(fc_record.keys(), fc_record.values()):
        val_sum = sum(cls)
        condition_entropy = condition_entropy + (val_sum/all_nums) * getEntropy(val_sum, cls)
    return condition_entropy
# 获得训练集，测试集，rate为test的占比，1-rate为train占比，直接一刀切，在此代码中，使用的那一步已经被注释掉，并未使用


def load_credits(matrix,rate):
    train_set, test_set = {}, {}
    edge = int(matrix.shape[0]*(1-rate))
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

    remain_index = list(set(indexes)-set(test_index))  # 所有索引中去掉已经选择的索引，将剩下的保留
    
    good_index, bad_index = [], []
    for index in remain_index:  # 统计剩下的样本中标签为1和不为1的样本索引
        if matrix[index, 0] == 1:
            good_index.append(index)
        else:
            bad_index.append(index)

    good_sample = random.sample(good_index, good_num)  # 标签为1的选出good_num个
    bad_sample = random.sample(bad_index, bad_num)  # 标签不为1的选出bad_num个
    samples = good_sample+bad_sample
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
        if matrix[index, 0] == 1:
            good_index.append(index)
        else:
            bad_index.append(index)
    
    good_train = random.sample(good_index, int(good_num*(1-test_ratio)))
    bad_train = random.sample(bad_index, int(bad_num*(1-test_ratio)))

    rest_good_sample = list(set(good_index)-set(good_train))
    rest_bad_sample = list(set(bad_index)-set(bad_train))

    good_test=random.sample(rest_good_sample, int(good_num*test_ratio))
    bad_test=random.sample(rest_bad_sample, int(bad_num*test_ratio))
    
    trains = good_train+bad_train
    tests = good_test+bad_test
    random.shuffle(trains)
    random.shuffle(tests)

    train_set['data'] = matrix[trains, 1:]
    train_set['target'] = matrix[trains, 0]
    test_set['data'] = matrix[tests, 1:]
    test_set['target'] = matrix[tests, 0]
    
    return train_set, test_set
# 联邦决策树分类器


class FederatedDecisionTreeClassifier:
    def __init__(self, dataset, all_feature, rank, client_num):
        # symbolize comm.rank()
        self.rank = rank
        self.client_num = client_num
        # dataset is the actual dataset a client owns
        self.dataset = dataset
        self.all_feature = all_feature
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
	# 计算最佳信息增益（？？跟决策树构建可能有关）

    def calculateBestInfoGain(self, dataset, features):
        all_nums = len(dataset['target'])
        class_nums = list(Counter(dataset['target']).values())
        best_info_gain, best_split_feature = float('-Inf'), None
        entropy = getEntropy(all_nums, class_nums)
        
        for col in range(dataset['data'].shape[1]):
            condition_entropy = getConditionEntropy(dataset, col, all_nums)
            info_gain = condition_entropy-entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_feature = features[col]
        return best_info_gain,best_split_feature
	# 判断feature[i]应该落在左节点还是右节点

    def repeated(self, features, i, val):
        if features[i] in self.threshold_map and val in self.threshold_map[features[i]]:
            return True
        return False
    # 计算最佳gini
    def calculateBestGini(self, indexes, features):
        rows, cols = len(indexes), len(features)
        best_gini_gain, split_feature, split_value = float('Inf'), -1, None
        # print('indexes:\n',indexes)
        features2 = [self.all_feature.index(i) for i in features]
        for i in range(cols):
            # medians = produce_median(list(set(self.dataset['data'][indexes, i])))
            newindex = np.lexsort((indexes,self.dataset['data'][indexes,features2[i]]))
            tmpind2 = Counter(self.dataset['target'][indexes])
            tmpind1 = Counter()
            for j in range(rows-1):
                val1 = self.dataset['data'][indexes[newindex[j]],features2[i]]
                val2 = self.dataset['data'][indexes[newindex[j+1]],features2[i]]
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
                gini_gain = p * result1 + (1-p) * result2
                if gini_gain < best_gini_gain :
                    best_gini_gain, split_feature = gini_gain, features2[i]
                    split_value = (val1+val2)/2

        # for i in range(cols):
        #     medians = produce_median(list(set(self.dataset['data'][indexes, i])))
        #     # print('medians:\n',medians)
        #     for val in medians:
        #         ind_1, ind_2 = splitIndexes(self.dataset, i, val, indexes)
        #         p = len(ind_1) / rows
        #         gini_gain = p*gini(self.dataset, ind_1)+(1-p)*gini(self.dataset, ind_2)
        #         if gini_gain < best_gini_gain and not self.repeated(features, i, val):
        #             best_gini_gain, split_feature, split_value = gini_gain, features[i], val
        
        return best_gini_gain, split_feature, split_value
            
    # 构建消息中的dest和message参数

    def getBestClient(self, message):  # buggy?!

        reply = {'sign': 'Success', 'feature': None, 'threshold': float('inf'), 'need_prune': False}
        dest = 0
        tmpGini = 9999999.9
        for i in range(1, len(message)):
            if message[i]['feature'] != -1 and message[i]['gini'] < tmpGini:
                reply['threshold'] = message[i]['value']
                reply['feature'] = message[i]['feature']
                dest = message[i]['rank']
                tmpGini = message[i]['gini']
                reply['need_prune'] = message[i]['need_prune']
        
        if reply['feature'] == -1 or reply['feature'] is None or reply['need_prune']:
            reply['sign'] = 'Prune'
        return dest, reply
    # 构建树

    def buildTree(self, indexes, rest_index, features):
        self.structure = self.FederatedTreeBuild(indexes, rest_index, features)
    
    # dataset will change through the split of tree，构建联邦树，rank=0表示中心

    def FederatedTreeBuild(self, indexes, rest_index, features):
        node = {'feature': None, 'threshold': None, 'gini': None, 'left': None, 'right': None}
        if len(set(self.dataset['target'][indexes])) == 1:
            # 本次划分中分类全部一致则生成叶子结点
            ind = indexes[0]
            node['class'] = self.dataset['target'][ind]
            return node
        if len(indexes)<4:
            node['class'] = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
            return node

        best_gini_gain, split_feature, split_value = None, None, None
        message = []
        if len(features) != 0 and self.rank != 0:
            best_gini_gain, split_feature, split_value = self.calculateBestGini(indexes, features)
            # 取得gini指数最小的特征,这里返回的局部feature

            # pre_pruning
            need_prune = False
            if split_feature is None or split_value is  None:
                print('what?')
            else:
                rest_left, rest_right = splitIndexes(self.dataset, split_feature, split_value, rest_index)
                left_indexes, right_indexes = splitIndexes(self.dataset, split_feature, split_value, indexes)
                if len(rest_left) != 0 and len(rest_right) != 0:
                    this_class = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
                    left_class = Counter(self.dataset['target'][left_indexes]).most_common(1)[0][0]
                    right_class = Counter(self.dataset['target'][right_indexes]).most_common(1)[0][0]
                    prune_acc = Counter(self.dataset['target'][rest_index])[this_class] / len(rest_index)
                    grow_acc = (Counter(self.dataset['target'][rest_left])[left_class] +
                                Counter(self.dataset['target'][rest_right])[right_class]) / len(rest_index)
                    if prune_acc >= grow_acc or prune_acc>0.9:
                        need_prune = True
            # end

            message = packMessage(best_gini_gain, split_feature, split_value, self.rank, need_prune)

            node['feature'], node['threshold'], node['gini'] = split_feature, split_value, best_gini_gain
        # MPI.Comm 类中的方法，https://www.cnblogs.com/zhbzz2007/p/5827059.html
        message = comm.gather(message, root=0)

        success_reply = None
        
        left_indexes, right_indexes = None, None
        is_selected = False
        prune_notice = None
        if self.rank == 0:
            destination, success_reply = self.getBestClient(message)
            # 选取给出最小gini的那个client
            if success_reply['sign'] == 'Prune':
                # success_reply['feature'] == -1
                prune_notice = success_reply
        
        prune_notice = comm.bcast(prune_notice if self.rank == 0 else None, root=0)
        if prune_notice is not None:
            node['class'] = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
            return node

        if self.rank == 0:
            req = comm.isend(success_reply, dest=destination, tag=1)
            req.wait()
            
            split_notice = comm.irecv(source=destination, tag=2)
            division_message = split_notice.wait()
            
            left_indexes = division_message['left_indexes']
            right_indexes = division_message['right_indexes']
            client_division = {
                'sign': 'Failure',
                'left_indexes': left_indexes,
                'right_indexes': right_indexes
            }

            split_notice = comm.irecv(source=destination, tag=2)
            division_message2 = split_notice.wait()

            rest_left = division_message2['rest_left']
            rest_right = division_message2['rest_right']
            client_division2 = {
                'sign': 'Failure',
                'rest_left': rest_left,
                'rest_right': rest_right
            }
            for i in range(1, self.client_num+1):
                if i != destination:
                    req = comm.isend(client_division, dest=i, tag=1)
                    req.wait()

                    req = comm.isend(client_division2, dest=i, tag=1)
                    req.wait()
            # print('master task over!')
        
        elif self.rank != 0:
            req = comm.irecv(source=0,tag=1)
            acknow = req.wait()
            if acknow['sign'] == 'Success':
                is_selected = True
                # column = features.index(acknow['feature'])
                left_indexes, right_indexes = splitIndexes(self.dataset, split_feature, split_value, indexes)
                rest_left, rest_right = splitIndexes(self.dataset, split_feature, split_value, rest_index)
                # reply = {'sign':'Success','feature':None,'threshold':float('inf')}
                # if acknow['feature'] not in self.threshold_map:
                #     self.threshold_map[acknow['feature']] = []
                # self.threshold_map[acknow['feature']].append(acknow['threshold'])

                # print('rank ',self.rank)
                # print('acknow',acknow)
                # print('left:\n',left_indexes)
                # print('right:\n',right_indexes)
                division_message ={
                    'sign': 'division',
                    'left_indexes': left_indexes,
                    'right_indexes': right_indexes
                }
                division_message2 = {
                    'sign': 'division',
                    'rest_left': rest_left,
                    'rest_right': rest_right
                }
                split_notice = comm.isend(division_message, dest=0, tag=2)
                split_notice.wait()

                split_notice = comm.isend(division_message2, dest=0, tag=2)
                split_notice.wait()

            elif acknow['sign'] == 'Failure':
                left_indexes, right_indexes = acknow['left_indexes'], acknow['right_indexes']

                req = comm.irecv(source=0, tag=1)
                acknow = req.wait()
                rest_left, rest_right = acknow['rest_left'], acknow['rest_right']

        if not is_selected:
            node['feature'], node['threshold'], node['gini'] = None, None, None

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            node = {'feature': None, 'threshold': None, 'gini': None, 'left': None, 'right': None}
            node['class'] = Counter(self.dataset['target'][indexes]).most_common(1)[0][0]
            return node

        print('left:',len(left_indexes))
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
        if self.rank != 0:
            if 'class' in node:
                self.leaves.append({'indexes': indexes, 'prediction': node['class']})
            elif node['feature'] is None:
                self.FederatedTreePredict(copy.deepcopy(indexes), node['left'])
                self.FederatedTreePredict(copy.deepcopy(indexes), node['right'])
            elif node['feature'] is not None:
                left_ind, right_ind = splitIndexes(self.tests, node['feature'], node['threshold'], indexes)
                self.FederatedTreePredict(left_ind, node['left'])
                self.FederatedTreePredict(right_ind, node['right'])

# 联邦森林分类器


class FederatedForestClassifier:
    def __init__(self, n_tree, rank, rate, frate, client):
        self.forest = []
        self.tree_num = n_tree
        self.rank = rank
        self.rate = rate
        self.frate = frate

        self.voters = None
        self.dataset = None
        self.all_feature = None
        self.tests = None
        if rank == 0:
            self.client_num = client
        else:
            self.client_num = None
        
    def fit(self, dataset, all_feature):
        self.dataset = dataset
        self.all_feature = all_feature
        data_num = self.dataset['data'].shape[0]
        feature_num = self.dataset['data'].shape[1]
        # self.voters = np.zeros((data_num,self.tree_num))
        
        for i in range(self.tree_num):
            if self.rank == 0:
                print('Tree ', i)
                print('--------------------------------------------------------------------------')
            selected_index = []
            rest_index = []
            if self.rank == 0:
                selected_index, rest_index = notify_selected_data(self.rate, data_num)  # 根据随机度从整体数据选出一定的行
                selected_features = notify_selected_feature(self.frate, feature_num)
            elif self.rank != 0:
                selected_index = None
                rest_index = None
                selected_features = None
            
            selected_index = comm.bcast(selected_index if self.rank == 0 else None, root=0)
            rest_index = comm.bcast(rest_index if self.rank == 0 else None, root=0)
            # 本次建树用的ID集合在所有client上一致,故广播
            selected_features = comm.scatter(selected_features if self.rank == 0 else None, root=0)
            # scatter分别传给不同client各自的特征
            fdtc = FederatedDecisionTreeClassifier(copy.deepcopy(self.dataset), copy.deepcopy(self.all_feature),
                                                   self.rank, self.client_num)
            
            fdtc.buildTree(selected_index, rest_index, selected_features)
            self.forest.append(fdtc)
            
    def predict(self, dataset):
        self.tests = dataset
        data_num = self.tests['data'].shape[0]
        self.voters = np.zeros((data_num, self.tree_num))
        for i in range(self.tree_num):
            classify_result = []
            if self.rank != 0:
                self.forest[i].predict(dataset)
                classify_result = self.forest[i].leaves
            
            classify_result = comm.gather(classify_result, root=0)
            
            if self.rank == 0:
                real_leafs = self.getUnion(classify_result)
                self.generateTarget(i, real_leafs)
        
        if self.rank == 0:
            self.bagging()
                
    def getUnion(self, result):
        real_leafs=[]
        leaf_num = len(result[1])
        client_num = self.client_num
        test_num = self.tests['data'].shape[0]
        
        for i in range(leaf_num):
            inter = set(list(x for x in range(test_num)))
            for j in range(1, client_num+1):
                inter = inter.intersection(set(result[j][i]['indexes']))
            # print('intersection ',i,':\n',inter)
            real_leafs.append({'indexes': list(inter), 'prediction': result[1][i]['prediction']})
        return real_leafs
    
    
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
                if abs(j - 1.0) < 0.00000001:
                    tmpsum += 1
            self.predict_prob[i] = tmpsum / len(self.voters[i, :])
        # print('final bagging\n',self.prediction)

    def getAccuracy(self, target):
        re, pred = find_p(self.predict_prob, target)
        return re, pred
        # # binaryAssessment(self.prediction,target)
        # all_predict, true_predict = 0, 0
        # class_0, class_1 = 0, 0
        # acc_0, acc_1 = 0, 0
        # assert(len(self.prediction) == len(target))
        # for i in range(len(self.prediction)):
        #     if target[i] == 0:
        #         class_0 = class_0+1
        #     elif target[i] == 1:
        #         class_1 = class_1+1
        #
        #     if self.prediction[i] == target[i]:
        #         if self.prediction[i] == 0:
        #            acc_0 = acc_0+1
        #         elif self.prediction[i] == 1:
        #             acc_1 = acc_1+1
        #         true_predict = true_predict+1
        #     all_predict = all_predict + 1
        # print('accuracy:', true_predict/all_predict)
        # print('0 accuracy:', acc_0/class_0)
        # print('1 accuracy:', acc_1/class_1)

# We represent 0 as the default server-rank
# self_rank means the rank of server-node
# comm_size means the rank of client-node


comm = MPI.COMM_WORLD
self_rank = comm.Get_rank()
comm_size = comm.Get_size()

# if self_rank == 0:
#     train_set,test_set= load_random(n_samples=800, centers=2, n_features=10,test_rate=0.2)

# train_set = comm.bcast(train_set if self_rank == 0 else None,root = 0)
# test_set = comm.bcast(test_set if self_rank == 0 else None,root = 0)
# test_set=copy.deepcopy(train_set)
hyperparams = {
    'client_num': comm_size - 1,
    'tree_num': 10,  # 树数量
    'rand_data_rate': 0.7,  # 随机度
    'rand_feature_rate': 0.9  # 特征随机度
}
datparams = {
    'data_num': 17461,
    'test_num': 3492,
    'feature_num': 100,
    'class_num': 2
}

train_set, test_set = None, None
if self_rank == 0:

    result_all = []
    pred_all = []

    train = pd.read_csv('train.csv', index_col=0)
    test = pd.read_csv('test.csv', index_col=0)
    train_set, test_set = {}, {}
    train_set['data'] = train.iloc[:, 1:].values
    train_set['target'] = train.iloc[:, 0].values
    test_set['data'] = test.iloc[:, 1:].values
    test_set['target'] = test.iloc[:, 0].values


    # csv_data = pd.read_csv('r0_12970+r1_6243__ff20+20.csv')
    # # train_set,test_set=load_credits(csv_data.values[:datparams['data_num']],rate=0.2)
    # # train_set,test_set=load_propotion_credits(csv_data.values,37500,2500,datparams['test_num'])
    # train_set, test_set = load_equal_credits(csv_data.values, 6243, 12970, 0.2)

train_sets = None
all_features = None
if self_rank == 0:
    train_sets, all_features = simulated_split_dataset(train_set)

train_set = comm.scatter(train_sets, root=0)
all_feature = comm.scatter(all_features, root=0)
test_set = comm.bcast(test_set if self_rank == 0 else None, root=0)

timw_start, time_end = None, None
if self_rank == 0:
    time_start = time.time()
    print("start time:",time_start)
ffc = FederatedForestClassifier(
    n_tree=hyperparams['tree_num'],
    rank=self_rank,
    rate=hyperparams['rand_data_rate'],
    frate=hyperparams['rand_feature_rate'],
    client=hyperparams['client_num']
)
ffc.fit(copy.deepcopy(train_set),copy.deepcopy(all_feature))
ffc.predict(test_set)
if self_rank == 0:
    time_end = time.time()
    re, pred = ffc.getAccuracy(test_set['target'])
    gt = pd.DataFrame(test_set['target'], index=test.index, columns=['GT'])
    gt['fed'] = pred
    gt.sort_index(inplace=True)
    gt.to_csv('./markov/FedForest_result_fed.csv')
    print('total time:', time_end-time_start)


# mpiexec -np 3 python FF_comm.py >> .\markov\mk_c2_.txt
