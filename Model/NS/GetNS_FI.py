import numpy as np
import warnings
import pandas as pd
from Model import params
from Utils.Utils import Utils,data

warnings.filterwarnings("ignore")



# get MPS based on the far item of the item
def Far_Item(rating_matrix, trainSet, K1, K2):

    # get every item's K negative items
    sim = np.corrcoef(rating_matrix.T)
    Sim_array = np.zeros((rating_matrix.shape[1], K2))
    NS_array = np.zeros((rating_matrix.shape[1], K1))
    for i in range(rating_matrix.shape[1]):
        NS_array[i] = sim[i].argsort()[:K1]
        Sim_array[i] = sim[i].argsort()[-K2:]

    NS_dict = {}
    for user, items in trainSet.items():
        sim_list, ns_lists = [], []                      # get the intersection_set of all items' negative items
        for i in items:

            ns_lists += (list(NS_array[i]))
            sim_list += (list(Sim_array[i]))

        ns_lists = set(ns_lists) - set(list(trainSet[user])) - set(sim_list)
        NS_dict[user] = list(ns_lists)
        # if len(ns_lists) == 0:
        #     print(user)
        print(user, len(ns_lists))

    return NS_dict



if __name__ == '__main__':
    
    dataset = params.dataset[-1]

    trainSet, interact_matrix, rating_matrix = data.loadTrainingData_modify(
        # "../../Data/"+dataset+"/initial/"+dataset+".base", params.Data[dataset]['split'],
        # "../../Data/" + dataset + "/final/RMZ_k300_r5_k61.base", params.Data[dataset]['split'],
        "../../Data/" + dataset + "/PS_N/N/10_k5_1.base", params.data[dataset]['split'],
            params.data[dataset]['userCount'],
            params.data[dataset]['itemCount'])
    testSet = data.loadTestData("../../Data/"+dataset+"/initial/"+dataset+".test", params.data[dataset]['split'])

    # far_item_nosim 20 100
    NS = Far_Item(rating_matrix, trainSet, 50, 50)
    # url = '../../Data/'+dataset+'/MPS/far_item/FI_nosim/50-50_1'
    url = '../../Data/' + dataset + '/final/PS_NS_50-50_1'
    Utils.saveNS_dict(NS, url+'.Data')

    print("Split train_set and test_set...")
    DF = pd.read_csv(url+'.Data', sep='\t', header=None, index_col=False)
    data.train_test_28(DF, url)


    print("It is over...")




