from sklearn.metrics import roc_auc_score
from math import sqrt
from sklearn.metrics import matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt

def binaryAssessment(prediction,target,p=0,n=1):  # 需要传入预测结果向量与实际结果向量,默认正例为0反例为1
    assert (len(prediction) == len(target))
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(prediction)):
        if prediction[i] == target[i]:
            if prediction[i] == p:
                TP += 1
            elif prediction[i] == n:
                TN += 1
        else:
            if prediction[i] == p:
                FP += 1
            elif prediction[i] == n:
                FN += 1
    print(TP,FP,TN,FN)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(TN+FP)
    FNR = FN/(TP+FN)
    Precision = TP/(TP+FP)
    ACC = (TP+TN)/(TP+FN+TN+FP)
    ErrorRate = (FP+FN)/(TP+FN+TN+FP)
    F1 = 2*Precision*TPR/(Precision+TPR)
    MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    AUC = roc_auc_score(prediction,target)
    # print('TPR/Recall: ',TPR)
    # print('TNR: ',TNR)
    # print('FPR: ',FPR)
    # print('FNR: ',FNR)
    print('ACC: ',ACC)
    print('Precision: ',Precision)
    # print('ErrorRate: ',ErrorRate)
    print('F1: ',F1)
    print('MCC: ',MCC)
    print('AUC: ',AUC)
    return [ACC,Precision,F1,MCC,AUC]

def find_p(y_prob, y_target):
    grid = np.multiply(0.01, range(100))
    # tune p
    p_list = []
    pred = [0] * len(y_prob)
    tmpp,tmpscore=0,-1
    for p in grid:
        for i in range(len(y_prob)):
            if y_prob[i] > p:
                pred[i] = 1
            else:
                pred[i] = 0
        score = matthews_corrcoef(y_target, pred)
        if score > tmpscore:
            tmpscore = score
            tmpp = p
        p_list.append(score)
    print(tmpp)
    for i in range(len(y_prob)):
        if y_prob[i] > tmpp:
            pred[i] = 1
        else:
            pred[i] = 0
    re = binaryAssessment(pred,y_target)
    return re, pred

def stability(data):
    chunk_size = int(len(data) / 10)
    stab = []
    for i in range(10):
        chunk = data.iloc[i*chunk_size:(i+1)*chunk_size]
        fed_right_sum = 0
        unfed_right_sum = 0
        for j in range(len(chunk)):
            if chunk.iloc[j,0] == chunk.iloc[j,1]:
                fed_right_sum += 1
            if chunk.iloc[j,0] == chunk.iloc[j,2]:
                unfed_right_sum += 1
        stab.append([fed_right_sum/chunk_size,unfed_right_sum/chunk_size])
    # print(stab)
    print('mean:',np.mean(stab,0))
    print('var:',np.var(stab,0))
    stab = np.array(stab)
    plt.figure()
    plt.plot(range(1, 11), stab[:, 0], 'b-s')
    plt.plot(range(1, 11), stab[:, 1], 'r-.s')
    plt.legend(['FedForest', 'RandomForest'])
    plt.title('Stability of FedForest & RandomForest')
    plt.xlabel('test set grouped by timing')
    plt.ylabel('group ACC')
    plt.ylim(0.5, 1.0)
    plt.show()

