import pandas as pd

dataset1 = pd.read_csv('.\markov\FedForest_result_fed.csv',index_col=0)
dataset2 = pd.read_csv('.\markov\FedForest_result_unfed.csv',index_col=0)
dataset = pd.concat([dataset1,dataset2.iloc[:,1]],axis=1)
dataset['time'] = None
# print(dataset)

reader = pd.read_csv('train_date.csv',index_col=0, iterator=True)
loop = True
chunkSize = 10000
chunks = []

flag = 0

while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        for i in range(len(chunk)):
            if flag == len(dataset):
                break
            if chunk.index[i] == dataset.index[flag]:
                print(flag,dataset.index[flag])
                tmp = chunk.iloc[i].dropna()
                dataset.iloc[flag,3] = max(tmp)
                flag += 1
        if flag == len(dataset):
            break
    except StopIteration:
        loop = False
        print("Iteration is stopped.")

dataset.sort_values(by=['time'],inplace=True)
dataset['status_fed'] = dataset['fed'] - dataset['GT']
dataset['status_unfed'] = dataset['unfed'] - dataset['GT']
dataset.to_csv('.\markov\FedForest_Markov.csv')
