import copy
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from Utils.Utils import Utils
from new_.Model import params, cfgan
from new_.Model.Fusion.epoch_10FPS_FNS import epoch_random_FPS1



def trainer(w, k, dataset, trainSet, testSet, trainVector, trainMaskVector, ns_rr, \
                TopN = [3, 5, 10, 20], epochCount = 500, save_model = False):
    pro_ZR = 50
    pro_PM = 50
    alpha = 0.2
    userCount = params.data[dataset]['userCount']
    itemCount = params.data[dataset]['itemCount']

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
    model_download = 0

    # trainVector_ = epoch_random_FPS.getPPS(trainVector.cpu().numpy(), r=5, k=50, k1=6)
    # trainVector_ = torch.Tensor(trainVector_)
    for epoch in range(epochCount):

        # trainVector_ = epoch_random_FPS.getPPS(trainVector.cpu().numpy(), r=5, k=50, k1=6)
        # trainVector_ = torch.Tensor(trainVector_)

        trainVector_ = copy.deepcopy(trainVector)

        if epoch > 150:
            trainVector_ = epoch_random_FPS1.Get_Modify_FPS(trainVector.cpu().numpy(),
                                                            trainSet, ns_rr, k, w, method=0)
            trainVector_ = epoch_random_FPS1.getPPS(trainVector_, r=1, k=9, k1=5)
            trainVector_ = torch.Tensor(trainVector_)

        # print(trainVector_[0,:50])

        #  Train Generator
        for step in range(G_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            realData = Variable(trainVector_[leftIndex:leftIndex + batchSize_G])
            eu = Variable(trainVector_[leftIndex:leftIndex + batchSize_G])

            # Select a random batch of negative items for every user
            n_items_pm, n_items_zr = Utils.select_negative_items(realData, itemCount,pro_PM, pro_ZR)
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

        #  Train Discriminator
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_D - 1)
            realData = Variable(trainVector_[leftIndex:leftIndex + batchSize_D])
            eu = Variable(trainVector_[leftIndex:leftIndex + batchSize_G])

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
                result_array = np.zeros((1, itemCount))
                result_array.astype(np.float)

                index = 0
                P, R, NDCG, MAP, MRR = [], [], [], [], []
                for testUser in testSet.keys():
                    data = Variable(trainVector_[testUser])
                    # #  Exclude the purchased vector that have occurred in the training set
                    result = G(data.reshape(1, itemCount)) + Variable(trainMaskVector[index])
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

                if save_model:
                    if topN == 0 and epoch >50:  # top5
                        if model_download < np.mean(P):
                            model_download = np.mean(P)
                        else:
                            torch.save({'G': G.state_dict()}, 'model_storage/' + save_model + '.pth')


            if topN == 0:
                print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{}'.format(epoch, epochCount,
                                                                                     d_loss.item(),
                                                                                     g_loss.item(),
                                                                                     np.mean(P)))



    return result_quotas


