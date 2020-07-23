import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
import random

dataset = pd.read_csv('HFL.csv', index_col=0)

y_train = dataset['Response']
X_train = dataset.drop(['Response'], axis=1)
X_train = X_train.fillna(-2)
# Start2
X_train1 = X_train.iloc[:,0:723]
X_train2 = X_train.iloc[:,723:]
X_train1 = X_train1.loc[:,X_train1.var()>0.001]
X_train2 = X_train2.loc[:,X_train2.var()>0.001]
pca1 = PCA(n_components=22)
X_train1 = pca1.fit_transform(X_train1)
pca2 = PCA(n_components=22)
X_train2 = pca2.fit_transform(X_train2)
X_train1 = pd.DataFrame(X_train1,index=dataset.index)
X_train2 = pd.DataFrame(X_train2,index=dataset.index)
dataset2 = pd.concat([y_train,X_train1,X_train2],axis=1)
print(dataset2)
dataset2.to_csv('VFL.csv')
