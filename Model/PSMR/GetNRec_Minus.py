import numpy
import numpy as np
import pandas as pd
import torch
from Model import params, cfgan
from torch.autograd import Variable
import copy
from Utils.Utils import Utils, data






def getNRec(site, k, R):
    Rec = np.array(pd.read_csv(site,header=None,index_col=False))
    ps_index = numpy.where(Rec < 0)
    Rec[ps_index] += 10000000
    NNS = Rec.argsort()[:k]

    NNS_R = []
    for u in range(NNS.shape[0]):
        for i in range(NNS.shape[1]):
            NNS_R.append([u,i,R])
    NNS_R_DF = pd.DataFrame(np.array(NNS_R))
    train_DF = pd.read_csv('../../Data/'+dataset+'/initial/'+dataset+'.base',header=None,
                           sep=params.data[dataset]['split']).iloc[:,:3]
    NNS_R_DF = pd.concat([NNS_R_DF,train_DF],ignore_index=True)

    NNS_R_DF.to_csv('../../Data/' + dataset + '/PS/MR/mr_1_k50.base',
                    sep=params.data[dataset]['split'], header=None, index=False)



if __name__ == '__main__':

    dataset = params.dataset[0]
    mode = params.modes[0]
    userCount = params.data[dataset]['userCount']
    itemCount = params.data[dataset]['itemCount']

    RecresultUrl = '../../Data/' + dataset + '/PS/rec_result.txt'

    K = []
    R = -1
    getNRec(RecresultUrl, 50, R)


