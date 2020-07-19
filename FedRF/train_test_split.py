import pandas as pd

csv_data = pd.read_csv('VFL.csv',index_col=0)

csv_data = csv_data.sample(frac=1)
num = int(len(csv_data)*0.8)
train = csv_data.iloc[:num]
test = csv_data.iloc[num:]
print(train.loc[train['Response']==0])
print(train.loc[train['Response']==1])
print(test.loc[test['Response']==0])
print(test.loc[test['Response']==1])
train.to_csv('train.csv')
test.to_csv('test.csv')

