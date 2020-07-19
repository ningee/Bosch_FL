import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics


class MarkovChain1:
    def __init__(self,data):
        self.data = pd.DataFrame(data)
        self.status = self.data.drop_duplicates().values.T[0]
        self.status_index = {}
        # for i in range(len(self.status)):
        #     self.status_index[self.status[i]] = i
        self.status_index[0] = 0
        self.status_index[1] = 1
        self.status_num = len(self.status)
        self.matrix = None

    def markov_matrix(self):
        if self.matrix is None:
            self.matrix = np.zeros((self.status_num,self.status_num))
            for i in range(len(self.data)-1):
                self.matrix[self.data.iloc[i],
                            self.data.iloc[i+1]] += 1
            # print(self.matrix)
            for i in range(len(self.matrix)):
                self.matrix[i] /= sum(self.matrix[i])
            # print(self.matrix)
            self.matrix = pd.DataFrame(self.matrix,index=[0,1],columns=[0,1])
        return self.matrix


class MarkovChain2:
    def __init__(self,data):
        self.data = pd.DataFrame(data)
        self.status = self.data.drop_duplicates().values.T[0]
        self.status_index = {}
        # for i in range(len(self.status)):
        #     self.status_index[self.status[i]] = i
        self.status_index['00'] = 0
        self.status_index['01'] = 1
        self.status_index['10'] = 2
        self.status_index['11'] = 3
        self.status_num = len(self.status)
        self.matrix = None
        self.matrix2 = None

    def markov_matrix(self):
        if self.matrix is None:
            self.matrix = np.zeros((4,4))
            for i in range(len(self.data)-3):
                self.matrix[self.data.iloc[i]*2+self.data.iloc[i+1],
                            self.data.iloc[i+2]*2+self.data.iloc[i+3]] += 1
            # print(self.matrix)
            for i in range(len(self.matrix)):
                self.matrix[i] /= sum(self.matrix[i])
            # print(self.matrix)
            self.matrix = pd.DataFrame(self.matrix,index=['00','01','10','11'],columns=['00','01','10','11'])
        return self.matrix

    def markov_matrix2(self):
        if self.matrix2 is None:
            self.matrix2 = np.zeros((4,4))
            for i in range(0,len(self.data)-3,2):
                self.matrix2[self.data.iloc[i]*2+self.data.iloc[i+1],
                            self.data.iloc[i+2]*2+self.data.iloc[i+3]] += 1
            # print(self.matrix)
            for i in range(len(self.matrix2)):
                self.matrix2[i] /= sum(self.matrix2[i])
            # print(self.matrix)
            self.matrix2 = pd.DataFrame(self.matrix2,index=['00','01','10','11'],columns=['00','01','10','11'])
        return self.matrix2


def distance(x,y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2

# # part 1: markov
# ff_result = pd.read_csv('./markov/FedForest_Markov.csv',index_col=0)
# random_list = pd.read_csv('./markov/random_list.csv',index_col=0)
#
# result_k1 = []
# result_k21 = []
# result_k22 = []
# for i in range(len(random_list)):
#     start = random_list.iloc[i,0]
#     L = random_list.iloc[i,1]
#     tmpMarkov = MarkovChain1(ff_result.iloc[start:start+L,0])
#     matrix1 = tmpMarkov.markov_matrix()
#     result_k1.append(matrix1.values.flatten())
#     tmpMarkov = MarkovChain2(ff_result.iloc[start:start+L,0])
#     matrix21 = tmpMarkov.markov_matrix()
#     matrix22 = tmpMarkov.markov_matrix2()
#     result_k21.append(matrix21.values.flatten())
#     result_k22.append(matrix22.values.flatten())
#
# result_k1 = np.array(result_k1)
# result_k21 = np.array(result_k21)
# result_k22 = np.array(result_k22)
#
# df_k1 = pd.DataFrame(result_k1)
# df_k21 = pd.DataFrame(result_k21)
# df_k22 = pd.DataFrame(result_k22)
# df_k1.to_csv('./markov/k1.csv')
# df_k21.to_csv('./markov/k21.csv')
# df_k22.to_csv('./markov/k22.csv')
# # end of part 1


df_k1 = pd.read_csv('./markov/k1.csv',index_col=0)
df_k21 = pd.read_csv('./markov/k21.csv',index_col=0)
df_k22 = pd.read_csv('./markov/k22.csv',index_col=0)


db = DBSCAN(eps=4,
                min_samples=2,
                metric=lambda a, b: distance(a, b)).fit(df_k1.values)
labels = db.labels_
print('每个样本的簇标号:')
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

print('分簇的数目: %d' % n_clusters_)
print("轮廓系数: %0.3f" % metrics.silhouette_score(df_k1.values, labels)) #轮廓系数评价聚类的好坏

for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = df_k1.values[labels == i]
    print(len(one_cluster))


db = DBSCAN(eps=8,
                min_samples=2,
                metric=lambda a, b: distance(a, b)).fit(df_k21.values)
labels = db.labels_
print('每个样本的簇标号:')
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

print('分簇的数目: %d' % n_clusters_)
print("轮廓系数: %0.3f" % metrics.silhouette_score(df_k21.values, labels)) #轮廓系数评价聚类的好坏

for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = df_k21.values[labels == i]
    print(len(one_cluster))


db = DBSCAN(eps=10,
                min_samples=5,
                metric=lambda a, b: distance(a, b)).fit(df_k22.values)
labels = db.labels_
print('每个样本的簇标号:')
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

print('分簇的数目: %d' % n_clusters_)
print("轮廓系数: %0.3f" % metrics.silhouette_score(df_k22.values, labels)) #轮廓系数评价聚类的好坏

for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = df_k22.values[labels == i]
    print(len(one_cluster))

# font0 = {'family' : 'Times New Roman',
#          'weight' : 'normal',
#          'size'   : 28,
# }
# font1 = {'family' : 'Times New Roman',
#          'weight' : 'normal',
#          'size'   : 23,
# }
# font2 = {'family' : 'Times New Roman',
#          'weight' : 'normal',
#          'size'   : 18,
# }

# plt.figure(figsize=(10.8, 5))
# plt.plot(result_k1[:, 0], 'b-s')
# plt.plot(result_k1[:, 1], 'b-.s')
# plt.plot(result_k1[:, 2], 'r-s')
# plt.plot(result_k1[:, 3], 'r-.s')
# plt.legend(['0->0','0->1','1->0','1->1'],prop=font2)
# plt.title('all parts of markov matrix (k=1)',font0)
# plt.xlabel('result on test set random selected',font1)
# plt.ylabel('ratio',font1)
# plt.ylim(0.0, 1.0)
# plt.savefig('D:\Works in BUAA\\federated\报告相关\\200517\\ff_markov1.png')
# plt.show()

# plt.figure(figsize=(10.8, 5))
# plt.plot(result_k21[:, 0], 'b-s')
# plt.plot(result_k21[:, 1], 'b-.s')
# plt.plot(result_k21[:, 2], 'b--s')
# plt.plot(result_k21[:, 3], 'b:s')
# plt.plot(result_k21[:, 4], 'r-s')
# plt.plot(result_k21[:, 5], 'r-.s')
# plt.plot(result_k21[:, 6], 'r--s')
# plt.plot(result_k21[:, 7], 'r:s')
# plt.plot(result_k21[:, 8], 'g-s')
# plt.plot(result_k21[:, 9], 'g-.s')
# plt.plot(result_k21[:, 10], 'g--s')
# plt.plot(result_k21[:, 11], 'g:s')
# plt.plot(result_k21[:, 12], 'y-s')
# plt.plot(result_k21[:, 13], 'y-.s')
# plt.plot(result_k21[:, 14], 'y--s')
# plt.plot(result_k21[:, 15], 'y:s')
# # plt.legend(['0->0','1->0'],prop=font2)
# plt.title('all parts of markov matrix (k=2)',font0)
# plt.xlabel('result on test set random selected',font1)
# plt.ylabel('ratio',font1)
# plt.ylim(0.0, 1.0)
# plt.savefig('D:\Works in BUAA\\federated\报告相关\\200517\\ff_markov21.png')
# plt.show()
#
# plt.figure(figsize=(10.8, 5))
# plt.plot(result_k22[:, 0], 'b-s')
# plt.plot(result_k22[:, 1], 'b-.s')
# plt.plot(result_k22[:, 2], 'b--s')
# plt.plot(result_k22[:, 3], 'b:s')
# plt.plot(result_k22[:, 4], 'r-s')
# plt.plot(result_k22[:, 5], 'r-.s')
# plt.plot(result_k22[:, 6], 'r--s')
# plt.plot(result_k22[:, 7], 'r:s')
# plt.plot(result_k22[:, 8], 'g-s')
# plt.plot(result_k22[:, 9], 'g-.s')
# plt.plot(result_k22[:, 10], 'g--s')
# plt.plot(result_k22[:, 11], 'g:s')
# plt.plot(result_k22[:, 12], 'y-s')
# plt.plot(result_k22[:, 13], 'y-.s')
# plt.plot(result_k22[:, 14], 'y--s')
# plt.plot(result_k22[:, 15], 'y:s')
# # plt.legend(['0->0','1->0'],prop=font2)
# plt.title('all parts of markov matrix (k=2)',font0)
# plt.xlabel('result on test set random selected',font1)
# plt.ylabel('ratio',font1)
# plt.ylim(0.0, 1.0)
# plt.savefig('D:\Works in BUAA\\federated\报告相关\\200517\\ff_markov22.png')
# plt.show()
