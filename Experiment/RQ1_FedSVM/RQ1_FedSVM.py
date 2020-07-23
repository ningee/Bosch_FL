'''
2020.05.01 - 2020.05.30
Guanghao Li
liguanghao@buaa.edu.cn
'''

import sys
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.datasets import load_digits

from svm import Federated_SVM, SVM

import pandas as pd
from sklearn.model_selection import train_test_split
from assessment import *
from markov import MarkovChain

FEDAVG = 1
FEDHIG = 2
FEDRAN = 3


def bosch_data(number_of_devices=4, algorithm=FEDAVG, num_eons=13,ran_seed=1):

    ds = pd.read_csv('./dataset/bosch/HFL.csv', index_col=0)
    ds = ds.loc[:, ds.var() > 0.001]
    ds = ds.fillna(-2)
    # print(ds.loc[ds['Response']==0])
    # print(ds.loc[ds['Response']==1])

    X = ds.drop(['Response'], axis=1)
    y = ds['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(0.8 * X.shape[0]))
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values

    gt = pd.DataFrame(y_test.iloc[500:].values,index=y_test.index[500:],columns=['GT'])
    # print(gt)

    re = [0, 0, 0, 0, 0]
    finalRe = [0, 0, 0, 0, 0]
    re_all = []
    finalPr = None

    for j in range(0,3):
        X_list = np.split(X_train[:int(len(X_train) / number_of_devices) * number_of_devices], number_of_devices)
        y_list = np.split(y_train[:int(len(y_train) / number_of_devices) * number_of_devices], number_of_devices)
        federation = Federated_SVM(
            X_test[:500],
            y_test.values[:500],
            X_test[500:],
            y_test.values[500:],
            global_aggregation=False,
            algorithm=algorithm,
            tol=0.0001)

        for i in range(number_of_devices):
            federation.add_participant(SVM(X_list[i], y_list[i], j, 1))

        re, pr = federation.run_eon(num_eons)  # n轮迭代训练过程,每轮获得全客户端的模型参数均值作为全局模型,并计算准确率
        # 随后全局模型参数直接用于下一轮各客户端的模型建立与训练

        re_all.append(re)
        if re[3] > finalRe[3]:
            finalRe = re
            finalPr = pr

    print("Final:")
    print('ACC: ', finalRe[0])
    print('Precision: ', finalRe[1])
    print('F1: ', finalRe[2])
    print('MCC: ', finalRe[3])
    print('AUC: ', finalRe[4])

    gt['fed'] = finalPr


    print('\nstart n=1:')

    for j in range(0,3):
        X_list = np.split(X_train[:int(len(X_train) / number_of_devices) * number_of_devices], number_of_devices)
        y_list = np.split(y_train[:int(len(y_train) / number_of_devices) * number_of_devices], number_of_devices)
        federation = Federated_SVM(
            X_test[:500],
            y_test.values[:500],
            X_test[500:],
            y_test.values[500:],
            global_aggregation=False,
            algorithm=algorithm,
            tol=0.0001)

        for i in range(1):
            federation.add_participant(SVM(X_list[i], y_list[i], j, 1))

        re, pr = federation.run_eon(1)  # n轮迭代训练过程,每轮获得全客户端的模型参数均值作为全局模型,并计算准确率
        # 随后全局模型参数直接用于下一轮各客户端的模型建立与训练


        re_all.append(re)
        if re[3] > finalRe[3]:
            finalRe = re
            finalPr = pr

    gt['unfed'] = finalPr
    print("Final:")
    print('ACC: ', finalRe[0])
    print('Precision: ', finalRe[1])
    print('F1: ', finalRe[2])
    print('MCC: ', finalRe[3])
    print('AUC: ', finalRe[4])

    gt.sort_index(inplace=True)
    gt['time'] = None

    reader = pd.read_csv('train_date.csv', index_col=0, iterator=True)
    loop = True
    chunkSize = 10000
    chunks = []
    flag = 0
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            for i in range(len(chunk)):
                if flag == len(gt):
                    break
                if chunk.index[i] == gt.index[flag]:
                    print(flag, gt.index[flag])
                    tmp = chunk.iloc[i].dropna()
                    gt.iloc[flag, 3] = max(tmp)
                    flag += 1
            if flag == len(gt):
                break
        except StopIteration:
            loop = False
            print("Iteration is stopped.")

    svm_result = gt
    svm_result.sort_values(by=['time'],inplace=True)
    svm_result['status_fed'] = svm_result['fed'] - svm_result['GT']
    svm_result['status_unfed'] = svm_result['unfed'] - svm_result['GT']
    svm_result.to_csv('svm_Markov.csv')


    svm_unfed_mk = MarkovChain(svm_result['status_unfed'])
    svm_fed_mk = MarkovChain(svm_result['status_fed'])
    print('ff_unfed_mk:\n', svm_unfed_mk.markov_matrix())
    print('ff_fed_mk:\n', svm_fed_mk.markov_matrix())

    stability(svm_result)

    return re, pr


bosch_data(number_of_devices=4,algorithm=1,
             num_eons=20,ran_seed=2333)
