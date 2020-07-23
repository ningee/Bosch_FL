import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
import random
from sklearn.tree import DecisionTreeClassifier


r1 = pd.read_csv('r1_all.csv', index_col=0)
r0 = pd.read_csv('r0_random36000.csv', index_col=0)

r0 = r0.drop_duplicates()
r1 = r1.drop_duplicates()
r0 = r0.loc[r0.count(axis=1)>100]
r1 = r1.loc[r1.count(axis=1)>100]


X = r1.iloc[:, :-1]
Y = r1.iloc[:,[-1]]
X = X.fillna(value=-2)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
df = pd.concat([pd.DataFrame(X,columns=['x','y']),pd.DataFrame(Y.values,columns=['Response'])],axis=1)
newR1 = pd.DataFrame()
for i in range(len(df)):
    if df.iloc[i,0]<10:
        newR1 = pd.concat([newR1,r1.iloc[[i]]],axis=0)
print(newR1)

X = r0.iloc[:, :-1]
Y = r0.iloc[:,[-1]]
X = X.fillna(value=-2)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
df = pd.concat([pd.DataFrame(X,columns=['x','y']),pd.DataFrame(Y.values,columns=['Response'])],axis=1)
newR0 = pd.DataFrame()
for i in range(len(df)):
    if df.iloc[i,0]<10:
        newR0 = pd.concat([newR0,r0.iloc[[i]]],axis=0)
print(newR0)

newR0 = newR0.sample(frac=1)
newR1 = newR1.sample(frac=1)
dataset = pd.concat([newR1.iloc[:6000],newR0.iloc[:12000]],axis=0)
print(dataset)
dataset.to_csv('HFL.csv')
