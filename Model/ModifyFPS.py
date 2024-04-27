import random

from Model import params
from Utils.Utils import Utils,data
import numpy as np
import pandas as pd


User_Trust = {}

def Get_FPS(trainSet, ns_rr, K):
    FPS = {}
    num = 0
    for u in trainSet.keys():
        u_ns = ns_rr[u].argsort()[-K:]
        fps = set(trainSet[u]) & set(u_ns)
        User_Trust[u] = len(fps)==0 and 1.0 or (1-len(fps)/len(trainSet[u]))
        num = num + len(fps)
        FPS[u] = { i:ns_rr[u,i] for i in fps}
    print(f'{K} FPS is ',num, num/len(trainSet))
    return FPS

def Modify(FPS, matrix, method):
    for u,i in FPS.items():
        items = list(i.keys())
        # print(u,i.values())
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


def Get_Modify_FPS(rating_matrix, trainSet, ns_rr, K, method=0):
    User_Trust = []
    FPS = []
    for u in trainSet.keys():
        u_ns = ns_rr[u].argsort()[-K:]
        fps = set(trainSet[u]) & set(u_ns)
        User_Trust.append([u, len(fps)==0 and 1.0 or (1-len(fps)/len(trainSet[u]))])
        for i in fps:
            FPS.append([u, i, ns_rr[u, i]])
    FPS = random.sample(FPS, int(0.2 * len(FPS)))
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
    return rating_matrix

if __name__ == '__main__':

    dataset = params.dataset[0]

    method = 2
    NS_method = params.methods['MPS'][2]  # the method to select negative samples

    K = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, ]
    # K_mod = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

    # ns_rr = np.array(pd.read_csv('../Data/' + dataset + '/final/intermediate/ns_rr_' + NS_method + '1.txt',
    #                              header=None, index_col=False))
    ns_rr = np.array(pd.read_csv('../Data/' + dataset + '/final/intermediate/ns_rr_neighbor.txt',
                                 header=None, index_col=False))
    trainSet, interact_matrix, rating_matrix = data.loadTrainingData_modify(
        '../Data/' + dataset + '/initial/' + dataset + '.base', params.data[dataset]['split'],
        # "../Data/" + dataset + "/PS_N/N/50_k6.base", params.Data[dataset]['split'],
        params.data[dataset]['userCount'],
        params.data[dataset]['itemCount'])
    testSet = data.loadTestData('../Data/' + dataset + '/initial/' + dataset + '.test', params.data[dataset]['split'])
    userList_test = list(testSet.keys())
    trainVector, trainMaskVector, batchCount = data.to_Vectors_trueRate(trainSet, params.data[dataset]['userCount'], \
                                                                        params.data[dataset]['itemCount'], userList_test, "userBased",
                                                                        rating_matrix)

    for k in [80]:
        url = '../Data/' + dataset + '/final/intermediate/' + NS_method + '/PS_NS'
        # url = '../Data/' + dataset + '/intermediate/' + NS_method + '/'

        FPSdict = Get_FPS(trainSet,ns_rr,k)
        Modify_matrix = Modify(FPSdict, rating_matrix, method)

        if method == 0:
            url = url + params.methods['FPS'][0] + '_mPS_k'
        if method == 1:
            url = url + params.methods['FPS'][1] + '_mPS_k'
        if method == 2:
            url = url + params.methods['FPS'][2] + '_mPS_k'
        Utils.saveNS(Modify_matrix, url + str(k) + '.base')

    # FPSdict = Get_FPS(trainSet, ns_rr, 300)
    # Modify_matrix = Modify(FPSdict, rating_matrix, method=0)
    # Utils.saveNS(Modify_matrix, '../Data/test/2.base')
    # a = Get_Modify_FPS(trainVector.cpu().numpy(), trainSet, ns_rr, 300, method=0)
    # Utils.saveNS(Modify_matrix, '../Data/test/1.base')

    print('It is over...')
