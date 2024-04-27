import numpy as np
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


def Get_FPS(User_Trust, trainSet, ns_rr, K):
    FPS = {}
    num = 0
    for u in trainSet.keys():
        u_ns = ns_rr[u].argsort()[-K:]
        fps = set(trainSet[u]) & set(u_ns)
        User_Trust[u] = len(fps)==0 and 1.0 or (1-len(fps)/len(trainSet[u]))
        num = num + len(fps)
        FPS[u] = { i:ns_rr[u,i] for i in fps}
    print(f'{K} FPS is ',num, num/len(trainSet))
    return FPS, User_Trust

def Modify(User_Trust, FPS, matrix, method):
    for u,i in FPS.items():
        items = list(i.keys())
        if len(items) != 0:
            if method == 0:         # Del
                matrix[u][items] = 0
            if method == 1:
                for item in items:          # RMZ
                    matrix[u][item] = (1-i[item])>0 and matrix[u][item] * (1 - i[item]) or 0
            if method == 2:
                for item in items:          # Trust
                    matrix[u][item] = matrix[u][item] * User_Trust[u]
    return matrix


def getPPS(rating_matrix, r, k, k1):
    sim = cosine_similarity(rating_matrix.T)
    topK = sim.argsort()[:,::-1]
    topK = topK[:,1:k+1]
    Nei_Num = []
    for u in range(rating_matrix.shape[0]):
        ps, ns = [], []
        for i in range(topK.shape[0]):
            ps.append(i) if rating_matrix[u,i] != 0 else ns.append(i)
        nei_num = []
        for i in ns:
            num = len(set(ps) & set(topK[i]))
            if num !=0 :
                nei_num.append((i,num))
        nei_num.sort(key=lambda x:x[1], reverse=True)
        nei_num = [i[0] for i in nei_num]
        rating_matrix[u, nei_num[:k1]] = r
        # Nei_Num.append(nei_num[:k1])
    return rating_matrix



def main(dataset, mode, k, train_url, target_url, userCount, itemCount):
    now_time = dt.datetime.now().strftime('%F %T')
    print(now_time, " running...")

    test_url = '../../../../Data/' + dataset + '/initial/' + dataset + '.test'
    train_url = train_url + '_mPS_k' + str(k) + '.base'
    target_url = target_url + mode + '_k' + str(k) + '.txt'

    testSet = data.loadTestData(test_url, params.data[dataset]['split'])
    trainSet, interact_matrix, rating_matrix = \
        data.loadTrainingData_modify(train_url, "\t", userCount, itemCount)
    userList_test = list(testSet.keys())

    if mode == "explicit":
        trainVector, trainMaskVector, batchCount = data.to_Vectors_trueRate(trainSet, userCount, \
                                                                            itemCount, userList_test, "userBased",
                                                                            rating_matrix)
    elif mode == "implicit":
        trainVector, trainMaskVector, batchCount = data.to_Vectors(trainSet, userCount, \
                                                                   itemCount, userList_test, "userBased")
    result_quotas = Trainer.trainer(dataset, testSet, \
                                trainVector, trainMaskVector)

    pd.DataFrame(result_quotas).to_csv(target_url)

    now_time = dt.datetime.now().strftime('%F %T')
    print(now_time, " end...")


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
    # userList_test = list(testSet.keys())
    # trainVector, trainMaskVector, batchCount = Data.to_Vectors(trainSet, userCount, \
    #                                                            itemCount, userList_test, "userBased")
    #
    #
    # # ----- 1 ------
    # Get_TNS_csv('../../Data/new_/' + dataset + '/TNS_50-50', rating_matrix, trainSet, 50, 50)
    # print("That getting TNS and spliting train_set and test_set is over...")
    #
    #
    # # ----- 2 ------
    # save_model = dataset + '_TNS_'
    # ns_rr = Trainer.trainer(dataset_id, testSet, trainVector, trainMaskVector, epochCount = 1000, save_model = save_model)
    # print("Training TNS to recommend TNS...")
    #
    # ns_rr = recommend_TNS(trainVector, save_model)
    # print("Recommend TNS...")
    #
    # # ----- 3 ------
    # User_Trust = {}
    method = 0
    NS_method = params.methods['MPS'][2]
    # K = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, ]
    # # K_mod = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # url = '../Data/' + dataset + '/final/intermediate/' + NS_method + '/'
    # for k in K:
    #     User_Trust = {}
    #     FPSdict, User_Trust = Get_FPS(User_Trust, trainSet,ns_rr,k)
    #     Modify_matrix = Modify(User_Trust, FPSdict, rating_matrix, method)
    #     if method == 0:
    #         url = url + params.methods['FPS'][0] + '_mPS_k'
    #     if method == 1:
    #         url = url + params.methods['FPS'][1] + '_mPS_k'
    #     if method == 2:
    #         url = url + params.methods['FPS'][2] + '_mPS_k'
    #     Utils.saveNS(Modify_matrix, url + str(k) + '.base')
    #
    #
    # # ----- 4 ------
    # train_data = pd.read_csv()



    # ----- 5 ------
    FPS_mm = params.methods['FPS'][0]
    train_url = '../../../../Data/' + dataset + '/intermediate/' + NS_method + '/' + FPS_mm
    target_url = '../../../../result_/' + dataset + '/' + NS_method + '_' + FPS_mm + '_'
    # p = Pool(processes=3)
    # for k in K:
        # for mode in params.modes:
    mode = params.modes[0]
    main(dataset, mode, 300, train_url, target_url, userCount, itemCount)
    # p.close()
    # p.join()


    print("It is over...")




