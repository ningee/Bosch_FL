import pandas as pd
import random

# 获取全数据中r==1的行
# 已完成
reader = pd.read_csv('train_numeric.csv',index_col=0, iterator=True)
loop = True
chunkSize = 100000
chunks = []
dr1 = pd.DataFrame()
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk.loc[chunk['Response']==1])
        dr1 = pd.concat([dr1, chunk.loc[chunk['Response']==1]])
    except StopIteration:
        loop = False
        print("Iteration is stopped.")

dr1.to_csv('./r1_all.csv')
# 已完成


# 获取全数据中一定数目r==0的行
# 已完成
reader = pd.read_csv('train_numeric.csv',index_col=0, iterator=True)
loop = True
chunkSize = 100000
chunks = []
dr0 = pd.DataFrame()
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        tmp = chunk.loc[chunk['Response']==0]
        tmp = tmp.sample(frac=1)
        # ranlist = random.sample(range(0, len(tmp)-1), 2000)
        # for i in ranlist:
        #     dr0 = pd.concat([dr0, tmp.iloc[[i]]], axis=0)
        dr0 = pd.concat([dr0, tmp.iloc[:3000]], axis=0)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
dr0.to_csv('./r0_random36000.csv')
# 已完成