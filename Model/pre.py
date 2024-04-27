import numpy as np
import pandas as pd
import torch
from Model import params
from Utils.Utils import Utils,data
from torch.autograd import Variable
import copy
import cfgan




if __name__ == '__main__':

    dataset = params.dataset[-1]
    mode = params.modes[0]

    userCount = params.data[dataset]['userCount']
    itemCount = params.data[dataset]['itemCount']
    trainSet, interact_matrix, rating_matrix = data.loadTrainingData_modify(
        # '../Data/'+dataset+'/MPS/far_item/FI_nosim/50-50_1.base', '\t', userCount, itemCount,)
        "../Data/" + dataset + "/final/PS_NS_50-50_1.base", '\t', userCount, itemCount, )
    testSet = data.loadTestData('../Data/'+dataset+'/initial/'+dataset+'.test', params.data[dataset]['split'])
    userList_test = list(testSet.keys())
    trainVector, trainMaskVector, batchCount = data.to_Vectors(trainSet, userCount, \
                                                               itemCount, userList_test, 'userBased')


    # read the paramters of G model, get the probability of  testuser
    G = cfgan.generator_no_userInfo(itemCount)
    state_dict = torch.load('model_storage/' + dataset + '_G_top5_neighbor_fusion.pth')
    G.load_state_dict(state_dict['G'])

    result_array = np.zeros((1, itemCount))
    result_array.astype(np.float)
    for trainUser in range(userCount):
        data = Variable(trainVector[trainUser])
        result = G(data.reshape(1, itemCount))
        result = result.reshape(1,itemCount)
        result = result.detach().numpy()
        result_array = np.concatenate([result_array, result])
    pd.DataFrame(result_array[1:,:]).to_csv('../Data/'+dataset+'/final/intermediate/ns_rr_neighbor.txt',
                                            header=None,index=False)


    print('It is over...')




