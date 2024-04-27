import random
import copy
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from Model import cfgan, params
import warnings
from Utils.Utils import Utils,data
import datetime as dt



warnings.filterwarnings("ignore")


def train_model(userCount,itemCount,testSet,trainVector,trainMaskVector,\
         TopN,epochCount,pro_ZR,pro_PM,alpha):
    
    # Build the generator and discriminator
    G=cfgan.generator_no_userInfo(itemCount)
    D=cfgan.discriminator(itemCount)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    G_step=5
    D_step=2
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
            n_items_pm,n_items_zr = Utils.select_negative_items(realData,itemCount,pro_PM,pro_ZR)
            ku_zp = Variable(torch.tensor(n_items_pm + n_items_zr))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku_zp
            
            # Generate a batch of new purchased vector
            fakeData=G(realData) 
            fakeData_ZP = fakeData * (eu + ku_zp)  
            fakeData_result=D(fakeData_ZP) 
            
            # Train the discriminator
            g_loss = np.mean(np.log(1.-fakeData_result.detach().numpy()+10e-5))  + alpha*regularization(fakeData_ZP,realData_zp)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()
            
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex=random.randint(1,userCount-batchSize_D-1)
            realData=Variable(trainVector[leftIndex:leftIndex+batchSize_D])
            eu = Variable(trainVector[leftIndex:leftIndex + batchSize_G])
            
            # Select a random batch of negative items for every user
            n_items_pm, _ = Utils.select_negative_items(realData,itemCount,pro_PM,pro_ZR)
            ku = Variable(torch.tensor(n_items_pm))
            
            # Generate a batch of new purchased vector
            fakeData=G(realData) 
            fakeData_ZP = fakeData * (eu + ku)
            
            # Train the discriminator
            fakeData_result=D(fakeData_ZP)
            realData_result=D(realData) 
            d_loss = -np.mean(np.log(realData_result.detach().numpy()+10e-5) + 
                              np.log(1. - fakeData_result.detach().numpy()+10e-5)) + 0*regularization(fakeData_ZP,realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
            
        if( epoch%1==0):
            for topN in range(len(TopN)):
                index = 0
                P, R, NDCG, MAP, MRR = [], [], [], [], []
                for testUser in testSet.keys():
                    data = Variable(trainVector[testUser])
                    # #  Exclude the purchased vector that have occurred in the training set
                    result = G(data.reshape(1,itemCount)) + Variable(trainMaskVector[index])
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

            # print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{}'.format(epoch, epochCount,
            #                                                                      d_loss.item(),
            #                                                                      g_loss.item(),
            #                                                                      np.mean(P)))

    return result_quotas


if __name__ == '__main__':

    for n in range(10):

        now_time = dt.datetime.now().strftime('%F %T')
        print(now_time, str(n)+" is running...")

        dataset = params.dataset[-1]

        TopN = [3, 5, 10, 20]
        epochs = 500
        pro_ZR = 50
        pro_PM = 50
        alpha = 0.1

        test_url = '../Data/' + dataset + '/initial/' + dataset + '.test'
        train_url = '../Data/' + dataset + '/final/RMZ_k300_r5_k6.base'    # RMZ_k300_r5_k6
        target_url = '../result_new/' + dataset + '/final/final'+str(n+2)+'.txt'

        userCount = params.data[dataset]['userCount']
        itemCount = params.data[dataset]['itemCount']
        testSet = data.loadTestData(test_url, params.data[dataset]['split'])
        trainSet, interact_matrix, rating_matrix = \
            data.loadTrainingData_modify(train_url, params.data[dataset]['split'], userCount, itemCount)
        userList_test = list(testSet.keys())

        trainSet1, interact_matrix1, rating_matrix1 = \
            data.loadTrainingData_modify('../Data/' + dataset + '/initial/' + dataset + '.base',
                                         params.data[dataset]['split'], userCount, itemCount)
        userList_test1 = list(testSet.keys())
        trainVector, trainMaskVector, batchCount = data.to_Vectors_trueRate(trainSet, userCount, \
                                itemCount, userList_test, "userBased", rating_matrix)
        trainVector1, trainMaskVector1, batchCount1 = data.to_Vectors(trainSet1, userCount, \
                                                                   itemCount, userList_test1, "userBased")

        result_quotas = train_model(userCount, itemCount, testSet, \
                                    trainVector, trainMaskVector1, TopN, epochs, pro_ZR, pro_PM, alpha)

        pd.DataFrame(result_quotas).to_csv(target_url)

        now_time = dt.datetime.now().strftime('%F %T')
        print(now_time, " It ended...")


