import pandas as pd
from collections import Counter
import numpy as np


class MarkovChain:
    def __init__(self,data):
        self.data = pd.DataFrame(data)
        self.status = self.data.drop_duplicates().values.T[0]
        self.status_index = {}
        # for i in range(len(self.status)):
        #     self.status_index[self.status[i]] = i
        self.status_index[0] = 0
        self.status_index[-1] = 1
        self.status_index[1] = 2
        self.status_num = len(self.status)
        self.matrix = None

    def markov_matrix(self):
        if self.matrix is None:
            self.matrix = np.zeros((self.status_num,self.status_num))
            for i in range(len(self.data)-1):
                self.matrix[self.status_index[self.data.iloc[i].values[0]],
                            self.status_index[self.data.iloc[i+1].values[0]]] += 1
            # print(self.matrix)
            for i in range(len(self.matrix)):
                self.matrix[i] /= sum(self.matrix[i])
            # print(self.matrix)
            self.matrix = pd.DataFrame(self.matrix,index=[0,-1,1],columns=[0,-1,1])
        return self.matrix
