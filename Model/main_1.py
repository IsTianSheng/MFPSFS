import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import cfgan
import warnings
from Model import params
from Utils.Utils import Utils,data

warnings.filterwarnings("ignore")



def main(userCount, itemCount, testSet, trainVector, trainMaskVector, \
         TopN, epochCount, pro_ZR, pro_PM, alpha):

    # Build the generator and discriminator
    G = cfgan.generator_no_userInfo(itemCount)
    D = cfgan.discriminator(itemCount)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    G_step = 5
    D_step = 2
    batchSize_G = 32
    batchSize_D = 32

    result_quotas = np.zeros([epochCount, 20])

    for epoch in range(epochCount):

        # ---------------------
        #  Train Generator
        # ---------------------

        for step in range(G_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            realData = Variable(trainVector[leftIndex:leftIndex + batchSize_G])
            eu = Variable(trainVector[leftIndex:leftIndex + batchSize_G])

            # Select a random batch of negative items for every user
            n_items_pm, n_items_zr = Utils.select_negative_items(realData, itemCount, pro_PM, pro_ZR)
            ku_zp = Variable(torch.tensor(n_items_pm + n_items_zr))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku_zp

            # Generate a batch of new purchased vector
            fakeData = G(realData)
            fakeData_ZP = fakeData * (eu + ku_zp)
            fakeData_result = D(fakeData_ZP)

            # Train the discriminator
            g_loss = np.mean(np.log(1. - fakeData_result.detach().numpy() + 10e-5)) + alpha * regularization(
                fakeData_ZP, realData_zp)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_D - 1)
            realData = Variable(trainVector[leftIndex:leftIndex + batchSize_D])
            eu = Variable(trainVector[leftIndex:leftIndex + batchSize_G])

            # Select a random batch of negative items for every user
            n_items_pm, _ = Utils.select_negative_items(realData, itemCount, pro_PM, pro_ZR)
            ku = Variable(torch.tensor(n_items_pm))

            # Generate a batch of new purchased vector
            fakeData = G(realData)
            fakeData_ZP = fakeData * (eu + ku)

            # Train the discriminator
            fakeData_result = D(fakeData_ZP)
            realData_result = D(realData)
            d_loss = -np.mean(np.log(realData_result.detach().numpy() + 10e-5) +
                              np.log(1. - fakeData_result.detach().numpy() + 10e-5)) + 0 * regularization(fakeData_ZP,
                                                                                                          realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        if (epoch % 1 == 0):
            for topN in range(len(TopN)):
                index = 0
                precisions = 0
                P, R, NDCG, MAP, MRR = [], [], [], [], []
                for testUser in testSet.keys():
                    data = Variable(trainVector[testUser])
                    result = G(data.reshape(1, itemCount)) + Variable(trainMaskVector[index])
                    # result = G(Data.reshape(1, itemCount)) + Variable(copy.deepcopy(trainMaskVector[index]))
                    result = result.reshape(itemCount)

                    p, r, ndcg, map, mrr = Utils.P_R_N_AP_RR(testSet[testUser], result, TopN[topN])
                    P.append(p)
                    R.append(r)
                    NDCG.append(ndcg)
                    MAP.append(map)
                    MRR.append(mrr)
                    index += 1

                result_quotas[epoch, 5 * topN + 0] = np.mean(P)
                result_quotas[epoch, 5 * topN + 1] = np.mean(R)
                result_quotas[epoch, 5 * topN + 2] = np.mean(NDCG)
                result_quotas[epoch, 5 * topN + 3] = np.mean(MAP)
                result_quotas[epoch, 5 * topN + 4] = np.mean(MRR)

            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{}'.format(epoch, epochCount,
                                                                                 d_loss.item(),
                                                                                 g_loss.item(),
                                                                                 np.mean(P)), precisions)

    return result_quotas


if __name__ == '__main__':

    TopN = [3, 5, 10, 20]
    epochs = 700
    pro_ZR = 50
    pro_PM = 50
    alpha = 0.1

    dataset = params.dataset[0]
    userCount = params.data[dataset]['userCount']
    itemCount = params.data[dataset]['itemCount']
    trainSet, interact_matrix, rating_matrix = \
        data.loadTrainingData_modify(params.data[dataset]['url'] + '.base',
                                     params.data[dataset]['split'], userCount, itemCount)
    testSet = data.loadTestData(params.data[dataset]['url'] + '.test',
                                params.data[dataset]['split'])
    userList_test = list(testSet.keys())

    trainVector, trainMaskVector, batchCount = data.to_Vectors_trueRate(trainSet, userCount, \
                                                              itemCount, userList_test, "userBased",rating_matrix)
    # trainVector, trainMaskVector, batchCount = Data.to_Vectors(trainSet, userCount, \
    #                                                            itemCount, userList_test, "userBased")

    a = trainVector.cpu().numpy()
    b = torch.from_numpy(a)

    result_quotas = main(userCount, itemCount, testSet, \
                         trainVector, trainMaskVector, TopN, epochs, pro_ZR, pro_PM, alpha)

    pd.DataFrame(result_quotas).to_csv('../result/' + dataset + '_standard_explicit2.txt')




