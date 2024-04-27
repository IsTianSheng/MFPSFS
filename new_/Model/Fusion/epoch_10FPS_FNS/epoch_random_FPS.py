import copy

import numpy as np
import random
import warnings
import pandas as pd
import torch
from Utils.Utils import Utils, data
from new_.Model import Trainer, params, cfgan
from torch.autograd import Variable
from multiprocessing import Pool
import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
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
        sim_list, ns_lists = [], []  # get the intersection_set of all items' negative items
        for i in items:
            ns_lists += (list(NS_array[i]))
            sim_list += (list(Sim_array[i]))
        ns_lists = set(ns_lists) - set(list(trainSet[user])) - set(sim_list)
        NS_dict[user] = list(ns_lists)
        print(user, len(ns_lists))

    return NS_dict

def Get_TNS_csv(url, rating_matrix, trainSet, K1, K2):
    NS = Far_Item(rating_matrix, trainSet, K1, K2)
    Utils.saveNS_dict(NS, url + '.Data')
    DF = pd.read_csv(url + '.Data', sep='\t', header=None, index_col=False)
    data.train_test_28(DF, url)


def recommend_TNS(trainVector, save_model):
    userCount = trainVector.shape[0]
    itemCount = trainVector.shape[1]
    G = cfgan.generator_no_userInfo(itemCount)
    state_dict = torch.load('model_storage/' + save_model + '.pth')
    G.load_state_dict(state_dict['G'])
    result_array = np.zeros((1, itemCount))
    result_array.astype(np.float)
    for trainUser in range(userCount):
        data = Variable(trainVector[trainUser])
        result = G(data.reshape(1, itemCount))
        result = result.reshape(1, itemCount)
        result = result.detach().numpy()
        result_array = np.concatenate([result_array, result])
    return result_array[1:,:]


def Get_Modify_FPS(rating_matrix_, trainSet, ns_rr, K, w, method=0):
    rating_matrix = copy.deepcopy(rating_matrix_)
    User_Trust = []
    FPS = []
    for u in trainSet.keys():
        u_ns = ns_rr[u].argsort()[-K:]
        fps = set(trainSet[u]) & set(u_ns)
        User_Trust.append([u, len(fps)==0 and 1.0 or (1-len(fps)/len(trainSet[u]))])
        for i in fps:
            FPS.append([u, i, ns_rr[u, i]])
    FPS = random.sample(FPS, int(w * len(FPS)))
    u_list = np.array(User_Trust)[:, 0]
    for line in FPS:
        if method == 0:      # Del
            rating_matrix[line[0]][line[1]] = 0
        if method == 1:         # RMZ
            rating_matrix[line[0]][line[1]] = (1 - line[2]) > 0 and rating_matrix[line[0]][line[1]] * (1 - line[2]) or 0
        if method == 2:         # Trust
            index = np.argwhere(u_list == line[0])
            Trust = User_Trust[index][1]
            rating_matrix[line[0]][line[1]] = rating_matrix[line[0]][line[1]] * Trust
    count1 = np.count_nonzero(rating_matrix)
    return rating_matrix


def getPPS(rating_matrix_, r, k, k1):
    rating_matrix = copy.deepcopy(rating_matrix_)
    count2 = np.count_nonzero(rating_matrix)
    sim = cosine_similarity(rating_matrix.T)
    topK = sim.argsort()[:,::-1]
    topK = topK[:,1:k+1]
    for u in range(rating_matrix.shape[0]):
        ps = np.where(rating_matrix[u] != 0)[0].tolist()
        ns = np.where(rating_matrix[u] == 0)[0].tolist()
        ns_topK = topK[ns]
        ps = set(ps)
        num = [*map(lambda x: len(ps & set(x)), ns_topK)]
        num = np.array(num)
        index = np.argsort(num).tolist()[::-1]
        fns = np.array(ns)[index[:k1]].tolist()
        rating_matrix[u, fns] = r
    count3 = np.count_nonzero(rating_matrix)
    return rating_matrix



def main(dataset, mode, w, k, train_url, target_url, userCount, itemCount, ns_rr):
    now_time = dt.datetime.now().strftime('%F %T')
    print(now_time, k, w, " running...")

    test_url = '../../../../Data/' + dataset + '/initial/' + dataset + '.test'
    # train_url = train_url + '_mPS_k' + str(k) + '.base'
    target_url = target_url + mode + '_k' + str(k) + '_w' + str(w) + '.txt'

    testSet = data.loadTestData(test_url, params.data[dataset]['split'])
    trainSet, interact_matrix, rating_matrix = \
        data.loadTrainingData_modify(train_url, "\t", userCount, itemCount)
    userList_test = list(testSet.keys())

    trainVector, trainMaskVector, batchCount = data.to_Vectors_trueRate(trainSet, userCount, \
                                                                        itemCount, userList_test, "userBased",
                                                                        rating_matrix)

    result_quotas = Trainer.trainer(w, k, dataset, trainSet, testSet, trainVector, trainMaskVector, ns_rr)

    pd.DataFrame(result_quotas).to_csv(target_url)

    now_time = dt.datetime.now().strftime('%F %T')
    print(now_time, k, w, " end...")


if __name__ == '__main__':
    dataset_id = 0
    dataset = params.dataset[dataset_id]
    userCount = params.data[dataset]['userCount']
    itemCount = params.data[dataset]['itemCount']
    # trainSet, interact_matrix, rating_matrix = Data.loadTrainingData_modify(
    #     "../../Data/"+dataset+"/initial/"+dataset+".base", params.Data[dataset]['split'],
    #     # "../../Data/" + dataset + "/final/RMZ_k300_r5_k61.base", params.Data[dataset]['split'],
    #     userCount, itemCount)
    # testSet = Data.loadTestData("../../Data/" + dataset + "/initial/" + dataset + ".test",
    #                             params.Data[dataset]['split'])


    method = 0
    NS_method = params.methods['MPS'][2]
    ns_rr = np.array(pd.read_csv('../../../../Data/' + dataset + '/final/intermediate/ns_rr_' + NS_method + '1.txt',
                                 header=None, index_col=False))


    # FPSdict, User_Trust = Get_Modify_FPS(rating_matrix, trainSet, ns_rr, 300, method)
    # Modify_matrix = Modify(User_Trust, FPSdict, rating_matrix, method)


    # ----- 5 ------
    FPS_mm = params.methods['FPS'][0]
    # train_url = '../../../../Data/' + dataset + '/intermediate/' + NS_method + '/' + FPS_mm
    train_url = '../../../../Data/' + dataset + '/initial/' + dataset + '.base'
    target_url = '../../../../result_/' + dataset + '/ablation/0' + NS_method + '_' + FPS_mm + '_'
    mode = params.modes[0]

    # main(dataset, mode, 300, train_url, target_url, userCount, itemCount, ns_rr)

    p = Pool(processes=3)
    for k in [800, 900]:
        # for mode in params.modes:
        for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            p.apply_async(main, (dataset, mode, w, k, train_url, target_url, userCount, itemCount, ns_rr))
    p.close()
    p.join()


    print("It is over...")




