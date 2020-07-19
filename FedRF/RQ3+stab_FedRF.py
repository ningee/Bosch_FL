from scipy.stats import ttest_ind
import pandas as pd
from assessment import *
from markov import *

ff_result = pd.read_csv('./markov/FedForest_Markov.csv',index_col=0)

binaryAssessment(ff_result.iloc[:,1].values,ff_result.iloc[:,0].values)
binaryAssessment(ff_result.iloc[:,2].values,ff_result.iloc[:,0].values)

ff_unfed_mk = MarkovChain(ff_result['status_unfed'])
ff_fed_mk = MarkovChain(ff_result['status_fed'])
print('ff_fed_mk:\n',ff_fed_mk.markov_matrix())
print('ff_unfed_mk:\n',ff_unfed_mk.markov_matrix())
matrix = abs(ff_unfed_mk.markov_matrix().values - ff_fed_mk.markov_matrix().values)
print('mean:',np.mean(matrix))
print('min:',np.min(matrix))
print('max:',np.max(matrix))
print('mid:',np.median(matrix))

print('stability:')
stability(ff_result)

# fed = {
#     'acc':      [0.422,0.450,0.500,0.481,0.499,0.571,0.561,0.567,0.555,0.509],
#     'precision':[0.657,0.657,0.673,0.657,0.656,0.652,0.664,0.673,0.668,0.509],
#     'f1':       [0.335,0.428,0.534,0.510,0.552,0.686,0.658,0.673,0.668,0.580],
#     'mcc':      [0.010,0.013,0.046,0.016,0.016,0.016,0.046,0.034,0.054,0.005],
#     'auc':      [0.506,0.507,0.522,0.508,0.507,0.508,0.522,0.517,0.526,0.005]
# }
#
# unfed = {
#     'acc':      [0.511,0.564,0.584,0.513,0.478,0.563,0.524,0.519],
#     'precision':[0.653,0.650,0.649,0.660,0.650,0.649,0.659,0.656],
#     'f1':       [0.581,0.677,0.708,0.577,0.511,0.677,0.599,0.593],
#     'mcc':      [0.011,0.009,0.008,0.027,0.005,0.005,0.027,0.020],
#     'auc':      [0.505,0.505,0.504,0.513,0.502,0.502,0.512,0.509]
# }
#
# frame_fed = pd.DataFrame(fed)
# frame_unfed = pd.DataFrame(unfed)
#
# for i in range(0,5):
#     t,p = ttest_ind(frame_fed.iloc[:,i],frame_unfed.iloc[:,i])
#     print(frame_fed.columns[i],':\tt=',t,", p=",p)
