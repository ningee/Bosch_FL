from assessment import *
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

N = 100
L_range = [300,1000]

ff_result = pd.read_csv('./markov/FedForest_Markov.csv',index_col=0)
num = len(ff_result)
fed_result = []
unfed_result = []
random_list = []

# for i in range(N):
# #     L = random.randint(L_range[0],L_range[1])
# #     start = random.randint(0, num - L)
# #     random_list.append([start,L])
# #     fed_re = binaryAssessment(ff_result.iloc[start:start+L,1].values,ff_result.iloc[start:start+L,0].values)
# #     unfed_re = binaryAssessment(ff_result.iloc[start:start+L,2].values,ff_result.iloc[start:start+L,0].values)
# #     fed_result.append(fed_re)
# #     unfed_result.append(unfed_re)
# #
# # fed_result = np.array(fed_result)
# # unfed_result = np.array(unfed_result)
# # random_list = np.array(random_list)
# # random_list = pd.DataFrame(random_list,columns=['start','L'])
# # random_list.to_csv('./markov/random_list.csv')

random_list = pd.read_csv('./markov/random_list.csv',index_col=0)
for i in range(N):
    start = random_list.iloc[i, 0]
    L = random_list.iloc[i,1]
    fed_re = binaryAssessment(ff_result.iloc[start:start + L, 1].values, ff_result.iloc[start:start + L, 0].values)
    unfed_re = binaryAssessment(ff_result.iloc[start:start+L,2].values,ff_result.iloc[start:start+L,0].values)
    fed_result.append(fed_re)
    unfed_result.append(unfed_re)
fed_result = np.array(fed_result)
unfed_result = np.array(unfed_result)

font0 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 23,
}
font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 20,
}
font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 16,
}
font3 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 16,
}

plt.figure(figsize=(10.8, 5))
plt.plot(fed_result[:, 0], 'b-s')
plt.plot(unfed_result[:, 0], 'r-.x')
plt.legend(['FedRF', 'RF'],prop=font2)
plt.title('ACC Values of FedRF and RF',font0)
plt.xlabel('random partial testing data group',font1)
plt.ylabel('ACC',font1)
plt.ylim(0.5, 1.0)
plt.savefig('./RQ2_result/ff1.png')
plt.show()

plt.figure()
tmpY = abs(fed_result[:, 0] - unfed_result[:, 0])
prop = np.array(np.arange(0.01,0.11,0.01))
for i in range(len(prop)):
    prop[i] = len(tmpY[tmpY<=prop[i]])/len(tmpY)
plt.bar(np.arange(0.01,0.11,0.01),prop,width=0.009)
plt.xlim(0,0.11)
plt.title('ACC Diff of FedRF and RF',font1)
plt.xlabel('Difference',font3)
plt.ylabel('Proportion',font3)
plt.savefig('./RQ2_result/ff1_1.png')
plt.show()

plt.figure(figsize=(10.8, 5))
plt.plot(fed_result[:, 1], 'b-s')
plt.plot(unfed_result[:, 1], 'r-.x')
plt.legend(['FedRF', 'RF'],prop=font2)
plt.title('Precision Values of FedRF and RF',font0)
plt.xlabel('random partial testing data group',font1)
plt.ylabel('Precision',font1)
plt.ylim(0.5, 1.0)
plt.savefig('./RQ2_result/ff2.png')
plt.show()

plt.figure()
tmpY = abs(fed_result[:, 1] - unfed_result[:, 1])
prop = np.array(np.arange(0.01,0.11,0.01))
for i in range(len(prop)):
    prop[i] = len(tmpY[tmpY<=prop[i]])/len(tmpY)
plt.bar(np.arange(0.01,0.11,0.01),prop,width=0.009)
plt.xlim(0,0.11)
plt.title('Precision Diff of FedRF and RF',font1)
plt.xlabel('Difference',font3)
plt.ylabel('Proportion',font3)
plt.savefig('./RQ2_result/ff2_1.png')
plt.show()

plt.figure(figsize=(10.8, 5))
plt.plot(fed_result[:, 2], 'b-s')
plt.plot(unfed_result[:, 2], 'r-.x')
plt.legend(['FedRF', 'RF'],prop=font2)
plt.title('F1 Values of FedRF and RF',font0)
plt.xlabel('random partial testing data group',font1)
plt.ylabel('F1',font1)
plt.ylim(0.5, 1.0)
plt.savefig('./RQ2_result/ff3.png')
plt.show()

plt.figure()
tmpY = abs(fed_result[:, 2] - unfed_result[:, 2])
prop = np.array(np.arange(0.01,0.11,0.01))
for i in range(len(prop)):
    prop[i] = len(tmpY[tmpY<=prop[i]])/len(tmpY)
plt.bar(np.arange(0.01,0.11,0.01),prop,width=0.009)
plt.xlim(0,0.11)
plt.title('F1 Diff of FedRF and RF',font1)
plt.xlabel('Difference',font3)
plt.ylabel('Proportion',font3)
plt.savefig('./RQ2_result/ff3_1.png')
plt.show()

plt.figure(figsize=(10.8, 5))
plt.plot(fed_result[:, 3], 'b-s')
plt.plot(unfed_result[:, 3], 'r-.x')
plt.legend(['FedRF', 'RF'],prop=font2)
plt.title('MCC Values of FedRF and RF',font0)
plt.xlabel('random partial testing data group',font1)
plt.ylabel('MCC',font1)
plt.ylim(0.0, 1.0)
plt.savefig('./RQ2_result/ff4.png')
plt.show()

plt.figure()
tmpY = abs(fed_result[:, 3] - unfed_result[:, 3])
prop = np.array(np.arange(0.02,0.21,0.02))
for i in range(len(prop)):
    prop[i] = len(tmpY[tmpY<=prop[i]])/len(tmpY)
plt.bar(np.arange(0.02,0.21,0.02),prop,width=0.018)
plt.xlim(0,0.22)
plt.title('MCC Diff of FedRF and RF',font1)
plt.xlabel('Difference',font3)
plt.ylabel('Proportion',font3)
plt.savefig('./RQ2_result/ff4_1.png')
plt.show()

plt.figure(figsize=(10.8, 5))
plt.plot(fed_result[:, 4], 'b-s')
plt.plot(unfed_result[:, 4], 'r-.x')
plt.legend(['FedRF', 'RF'],prop=font2)
plt.title('AUC Values of FedRF and RF',font0)
plt.xlabel('random partial testing data group',font1)
plt.ylabel('AUC',font1)
plt.ylim(0.4, 1.0)
plt.savefig('./RQ2_result/ff5.png')
plt.show()

plt.figure()
tmpY = abs(fed_result[:, 4] - unfed_result[:, 4])
prop = np.array(np.arange(0.01,0.11,0.01))
for i in range(len(prop)):
    prop[i] = len(tmpY[tmpY<=prop[i]])/len(tmpY)
plt.bar(np.arange(0.01,0.11,0.01),prop,width=0.009)
plt.xlim(0,0.11)
plt.title('AUC Diff of FedRF and RF',font1)
plt.xlabel('Difference',font3)
plt.ylabel('Proportion',font3)
plt.savefig('./RQ2_result/ff5_1.png')
plt.show()
