import copy
import random
import numpy as np
import warnings
import pandas as pd
import torch
from OMRF import params, cfgan
from OMRF.model import Trainer
from Utils.Utils import Utils, data
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
        # print(user, len(ns_lists))

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
    state_dict = torch.load(save_model)
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
    # User_Trust = []
    User_Trust = {}
    FPS = []
    for u in trainSet.keys():
        u_ns = ns_rr[u].argsort()[-K:]
        fps = set(trainSet[u]) & set(u_ns)
        # User_Trust.append([u, len(fps)==0 and 1.0 or (1-len(fps)/len(trainSet[u]))])
        User_Trust[u] = len(fps) == 0 and 1.0 or (1 - len(fps) / len(trainSet[u]))
        for i in fps:
            FPS.append([u, i, ns_rr[u, i]])
    FPS = random.sample(FPS, int(w * len(FPS)))
    # u_list = np.array(User_Trust)[:, 0]
    for line in FPS:
        if method == 0:      # Del
            rating_matrix[line[0]][line[1]] = 0
        if method == 1:         # RMZ
            rating_matrix[line[0]][line[1]] = (1 - line[2]) > 0 and rating_matrix[line[0]][line[1]] * (1 - line[2]) or 0
        if method == 2:         # Trust
            rating_matrix[line[0]][line[1]] = rating_matrix[line[0]][line[1]] * User_Trust[line[0]]
            # index = np.argwhere(u_list == line[0])
            # Trust = User_Trust[index][1]
            # rating_matrix[line[0]][line[1]] = rating_matrix[line[0]][line[1]] * Trust
    return rating_matrix


def getPPS(rating_matrix_, r, k, k1):
    rating_matrix = copy.deepcopy(rating_matrix_)
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
    return rating_matrix



def main(dataset, train_url, target_url, userCount, itemCount):
    now_time = dt.datetime.now().strftime('%F %T')
    print(now_time, '     ',target_url, " is running...")

    test_url = '../Data/' + dataset + '/initial/' + dataset + '.test'
    testSet = data.loadTestData(test_url, params.data[dataset]['split'])
    trainSet, interact_matrix, rating_matrix = \
        data.loadTrainingData_modify(train_url, ",", userCount, itemCount)
    userList_test = list(testSet.keys())

    trainVector, trainMaskVector, batchCount = data.to_Vectors_trueRate(trainSet, userCount, \
                             itemCount, userList_test, "userBased", rating_matrix)

    result_quotas = Trainer.trainer(dataset, testSet, trainVector, trainMaskVector)

    pd.DataFrame(result_quotas).to_csv(target_url)

    now_time = dt.datetime.now().strftime('%F %T')
    print(now_time, target_url, " is end...")



if __name__ == '__main__':
    dataset_id = -1
    dataset = params.dataset[dataset_id]
    userCount = params.data[dataset]['userCount']
    itemCount = params.data[dataset]['itemCount']

    for i in range(10):
        count = i+10

        # trainSet, interact_matrix, rating_matrix_tomodify = 0, 0, 0
        # if count == 1:
        #     trainSet, interact_matrix, rating_matrix_tomodify = data.loadTrainingData_modify(
        #         '../../Data/' + dataset + '/final/Trust_k80_r1_k51.base', params.data[dataset]['split'],
        #         userCount, itemCount)
        # else:
        #     trainSet, interact_matrix, rating_matrix_tomodify = data.loadTrainingData_modify(
        #         '../Data/' + dataset + '/FNS/r1_k10_k5_No'+str(count-1)+'.base', params.data[dataset]['split'],
        #         userCount, itemCount)
        #
        #
        # # ----- 1 ------
        # now_time = dt.datetime.now().strftime('%F %T')
        # print(now_time, "     1 ---- Split TNS train_set and test_set...")
        # NS = Far_Item(rating_matrix_tomodify, trainSet, 50, 50)
        # url_TNS = '../Data/' + dataset + '/FPS/TNS/50-50_' + 'No' + str(count)
        # Utils.saveNS_dict(NS, url_TNS + '.data')
        # DF = pd.read_csv(url_TNS + '.data', sep='\t', header=None, index_col=False)
        # data.train_test_28(DF, url_TNS)
        #
        #
        # # ----- 2 ------
        # now_time = dt.datetime.now().strftime('%F %T')
        # print(now_time, "     2 ---- Training TNS to recommend TNS...")
        # save_model = '../Data/' + dataset + '/FPS/model/TNS_' + 'No' + str(count) + '.pth'
        # trainSet, interact_matrix, rating_matrix = data.loadTrainingData_modify(
        #     url_TNS + '.base', '\t', userCount, itemCount, )
        # testSet_TNS = data.loadTestData(url_TNS + '.test', '\t')
        # userList_test_TNS = list(testSet_TNS.keys())
        # trainVector_TNS, trainMaskVector_TNS, batchCount = data.to_Vectors(trainSet, userCount, \
        #                                                            itemCount, userList_test_TNS, "userBased")
        # ns_rr = Trainer.trainer(dataset, testSet_TNS, trainVector_TNS, trainMaskVector_TNS,
        #                         epochCount = 1000, save_model = save_model)
        # ns_rr = recommend_TNS(trainVector_TNS, save_model)
        #
        #
        # # ----- 3 ------
        # now_time = dt.datetime.now().strftime('%F %T')
        # print(now_time, "     3 ---- Modify FPS-ratings...")
        # method = 2
        # url_FPS = '../Data/' + dataset + '/FPS/' + 'No' + str(count)+'_'\
        #           + params.methods['FPS'][2] + '_mPS_k' + '80.base'
        # Modify_matrix = Get_Modify_FPS(rating_matrix_tomodify, trainSet, ns_rr, 80, 1, method)
        # Utils.saveNS(Modify_matrix, url_FPS)
        #
        #
        #
        # # ----- 4 ------
        # now_time = dt.datetime.now().strftime('%F %T')
        # print(now_time, "     4 ---- Add FNS by modified ratings_matrix...")
        # train_url_FNS = url_FPS
        # trainSet_FNS, interact_matrix_FNS, rating_matrix_FNS = \
        #     data.loadTrainingData_modify(train_url_FNS, ',', userCount, itemCount)
        # rating_matrix_FNS = getPPS(rating_matrix_FNS, r=1, k=10, k1=5)
        # url_FNS = '../Data/' + dataset + '/FNS/r1_k10_k5_' + 'No' + str(count) + '.base'
        # Utils.saveNS(rating_matrix_FNS, url_FNS)



        # ----- 5 ------
        p = Pool(processes=2)

        # FPS
        url_FPS = '../Data/' + dataset + '/FPS/' + 'No' + str(count)+'_' + params.methods['FPS'][2] + '_mPS_k' + '80.base'
        target_url_FPS = '../result/' + dataset + '/FPS/' + 'No' + str(count)+'_' \
                         + params.methods['FPS'][2] + '_mPS_k80.txt'
        p.apply_async(main, (dataset, url_FPS, target_url_FPS, userCount, itemCount))

        # FNS
        url_FNS = '../Data/' + dataset + '/FNS/r1_k10_k5_' + 'No' + str(count) + '.base'
        target_url_FNS = '../result/' + dataset + '/FNS/r1_k10_k5_' + 'No' + str(count) + '.txt'
        p.apply_async(main, (dataset, url_FNS, target_url_FNS, userCount, itemCount))


        p.close()
        p.join()




    print("It is over...")




