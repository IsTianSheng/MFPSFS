import matplotlib
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



class Utils:

    def saveNS_dict(NSdict, site):
        userlist, itemlist = [], []
        for u,i in NSdict.items():
            userlist += [u+1]*len(i)
            itemlist += i
        itemlist = [i+1 for i in itemlist]
        ratelist = [1]*len(userlist)
        pd.DataFrame({'userlist':userlist,'itemlist':itemlist,'ratelist':ratelist})\
            .to_csv(site, sep='\t', header=None, index=False)

    def saveNS(NSarray,site):
        locatioin = np.nonzero(NSarray)
        locatioin_1 = tuple(arr+1 for arr in locatioin)
        NS_file = np.vstack([locatioin_1,NSarray[locatioin]])
        NS_file = NS_file.T
        pd.DataFrame(NS_file.astype(np.float64)).to_csv(site, sep=',', header=None, index=False)

    def select_negative_items(realData, itemCount, num_pm, num_zr):
        data = np.array(realData)
        n_items_pm = np.zeros_like(data)
        n_items_zr = np.zeros_like(data)
        for i in range(data.shape[0]):
            # p_items = np.where(Data[i] != 0)[0]
            # all_item_index = random.sample(range(Data.shape[1]), itemCount)
            # for j in range(p_items.shape[0]):
            #     all_item_index.remove(list(p_items)[j])
            all_item_index = np.where(data[i] == 0)[0]
            random.shuffle(all_item_index)

            n_item_index_pm = all_item_index[0: num_pm]
            n_item_index_zr = all_item_index[num_pm: (num_pm + num_zr)]
            n_items_pm[i][n_item_index_pm] = 1
            n_items_zr[i][n_item_index_zr] = 1

        return n_items_pm, n_items_zr

    def P_R_N_AP_RR(trued, pred, topN):
        pred = pred.tolist()
        pred = np.array(pred).argsort()[::-1][:topN]
        pred = list(pred)
        p,r,ndcg,ap,rr = 0.0,0.0,0.0,0.0,0.0
        TP = list([i for i in pred if i in trued])

        p = float(len(TP)) / len(pred)
        r = float(len(TP)) / len(trued)
        if len(TP) != 0:
            TP_index = np.array([1 + pred.index(i) for i in TP])
            DCG = 1 / np.log2(TP_index + 1)
            IDCG = 1 / np.log2(np.arange(len(TP)) + 1 + 1)
            ndcg = sum(DCG) / sum(IDCG)
            ap = sum([(list(TP_index).index(i) + 1) / i for i in TP_index]) / topN
            rr = np.mean(1 / TP_index)
        return p, r, ndcg, ap, rr

    def GetPR(trued, pred, start, interval):
        pred = list(pred)
        pred = np.array(pred).argsort()[::-1][start:start+interval]
        pred = list(pred)
        p, r = 0.0, 0.0
        TP = list([i for i in pred if i in trued])
        p = float(len(TP)) / len(pred)
        r = float(len(TP)) / len(trued)
        return p, r



class data:

    def train_test_28(DataFrame,url):
        train,test = train_test_split(DataFrame, test_size= 0.2 , random_state=0)
        train.to_csv(url+".base", sep="\t", header=None, index=False)
        test.to_csv(url+".test", sep="\t", header=None, index=False)

    def loadTrainingData_modify(trainFile, splitMark, userCount, itemCount,):
        trainSet = defaultdict(list)
        max_u_id = -1
        max_i_id = -1

        for line in open(trainFile):
            data = line.strip().split(splitMark)
            userId = int(float(data[0])) - 1
            itemId = int(float(data[1])) - 1
            trainSet[userId].append(itemId)
            max_u_id = max(userId, max_u_id)
            max_i_id = max(itemId, max_i_id)

        # To get two matrixes
        interact_matrix, rating_matrix = np.zeros((userCount, itemCount)), np.zeros((userCount, itemCount))
        for line in open(trainFile):
            data = line.strip().split(splitMark)
            userId = int(float(data[0])) - 1
            itemId = int(float(data[1])) - 1
            interact_matrix[userId][itemId] = 1
            # print(Data)
            rating_matrix[userId][itemId] = float(data[2])

        # print("Training Data loading done")
        return trainSet,  interact_matrix, rating_matrix

    def loadTestData(testFile, splitMark):
        testSet = defaultdict(list)
        # max_u_id = -1
        # max_i_id = -1
        for line in open(testFile):
            data = line.strip().split(splitMark)
            userId = int(float(data[0])) - 1
            itemId = int(float(data[1])) - 1
            testSet[userId].append(itemId)
            # max_u_id = max(userId, max_u_id)
            # max_i_id = max(itemId, max_i_id)
        # print("Test Data loading done")
        return testSet

    def to_Vectors(trainSet, userCount, itemCount, userList_test, mode):
        testMaskDict = defaultdict(lambda: [0] * itemCount)
        batchCount = userCount  # ¸Ä¶¯  Ö±½ÓÐ´³ÉuserCount
        if mode == "itemBased":  # ¸Ä¶¯  itemCount userCount»¥»»   batchCountÊÇÎïÆ·Êý
            userCount = itemCount
            itemCount = batchCount
            batchCount = userCount
        trainDict = defaultdict(lambda: [0] * itemCount)
        for userId, i_list in trainSet.items():
            for itemId in i_list:
                testMaskDict[userId][itemId] = -99999
                # trainDict[userId][itemId] = 1.0
                if mode == "userBased":
                    trainDict[userId][itemId] = 1.0
                else:
                    trainDict[itemId][userId] = 1.0
        trainVector = []
        for batchId in range(batchCount):
            trainVector.append(trainDict[batchId])
        testMaskVector = []
        for userId in userList_test:
            testMaskVector.append(testMaskDict[userId])
        # for i in range(userCount):
        #     if i not in userList_test:
        #         testMaskVector.append([0] * itemCount)
        #     else:
        #         testMaskVector.append(testMaskDict[userId])

        return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount

    def to_Vectors_trueRate(trainSet, userCount, itemCount, userList_test, mode, rating_matrix):
        testMaskDict = defaultdict(lambda: [0] * itemCount)
        batchCount = userCount  # ¸Ä¶¯  Ö±½ÓÐ´³ÉuserCount
        if mode == "itemBased":  # ¸Ä¶¯  itemCount userCount»¥»»   batchCountÊÇÎïÆ·Êý
            userCount = itemCount
            itemCount = batchCount
            batchCount = userCount
        trainDict = defaultdict(lambda: [0] * itemCount)
        for userId, i_list in trainSet.items():
            for itemId in i_list:
                testMaskDict[userId][itemId] = -99999
                if mode == "userBased":
                    trainDict[userId][itemId] = rating_matrix[userId, itemId]
                else:
                    trainDict[itemId][userId] = rating_matrix[userId, itemId]
        trainVector = []
        for batchId in range(batchCount):
            trainVector.append(trainDict[batchId])

        testMaskVector = []
        for userId in userList_test:
            testMaskVector.append(testMaskDict[userId])
        #    print("Converting to vectors done....")
        return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount



class Graph:

    # DataList: dict --->   {'label':[Data],....}
    def Graph_one_(DataDict, X_axis, Y_axis, ylabel, title='Result Anaylize', url=False):
        matplotlib.rc("font", family='Microsoft YaHei')
        LineColor = ['#FF0000', '#6495ED', '#808000', '#3CB371', '#ed7d31', '#FFF1E0']  # the firset is ours
        DataList = list(DataDict.values())
        LabelList = list(DataDict.keys())

        for i in range(len(DataDict)):
            plt.plot(X_axis, DataList[i], color=LineColor[i], linestyle='-', marker='o', alpha=0.5,
                     linewidth=1, label=LabelList[i])

        plt.legend()  # ÏÔÊ¾ÉÏÃæµÄlabel
        # plt.scatter()
        plt.xlabel('K2')
        plt.ylabel(ylabel)
        # plt.title(title)
        plt.grid(visible='True', axis='y')  # Ö»ÏÔÊ¾Íø¸ñºáÏß
        plt.ylim(Y_axis[0], Y_axis[1])  # ½öÉèÖÃyÖá×ø±ê·¶Î§
        if url:
            plt.savefig(url + 'png')
        plt.show()

    # DataList: dict --->   {'label':[Data],....}
    def Graph_one(DataDict, X_axis, Y_axis, ylabel, title='Result Anaylize', url=False):
        matplotlib.rc("font", family='Microsoft YaHei')
        # LineColor = ['#f74d4d', '#0c84c6', '#ffa510', '#70ad47', '#ed7d31', '#3682be']    # the firset is ours
        # LineColor = ['#FF0000', '#4169E1', '#FFFF00', '#70ad47', '#ed7d31', '#CC6633']  # the firset is ours
        LineColor = ['#FF0000', '#6495ED', '#808000', '#3CB371', '#ed7d31', '#FFF1E0']  # the firset is ours
        DataList = list(DataDict.values())
        LabelList = list(DataDict.keys())

        for i in range(len(DataDict)):
            plt.plot(X_axis, DataList[i], color=LineColor[i], linestyle='-', alpha=0.5,
                     linewidth=1, label=LabelList[i])

        plt.legend()  # ÏÔÊ¾ÉÏÃæµÄlabel
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        # plt.title(title)

        plt.grid(visible='True', axis='y')  # Ö»ÏÔÊ¾Íø¸ñºáÏß

        plt.ylim(Y_axis[0], Y_axis[1])  # ½öÉèÖÃyÖá×ø±ê·¶Î§
        if url:
            plt.savefig(url+'png')
        plt.show()


    # DataList: dict --->   {'label':[Data],....}
    def Graph_one_NS(DataDict, X_axis, Y_axis, ylabel, title='Result Anaylize'):
        matplotlib.rc("font", family='Microsoft YaHei')
        LineColor = ['#FF0000', '#6495ED', '#FFA500', '#808000', '#3CB371', '#ed7d31']  # the firset is ours
        LineMarker = ['o', '*', '^', 'x', ',']
        DataList = list(DataDict.values())
        LabelList = list(DataDict.keys())
        X_axis_len = range(len(X_axis))

        for i in range(len(DataDict)):
            plt.plot(X_axis_len, DataList[i], color=LineColor[i], linestyle='-', alpha=0.5, linewidth=1.5,
                     marker=LineMarker[i], label=LabelList[i])

        plt.legend()  # ÏÔÊ¾ÉÏÃæµÄlabel
        plt.xticks(X_axis_len, X_axis, rotation=45)
        plt.xlabel('TopN')
        plt.ylabel(ylabel)
        plt.title(title)

        plt.grid(visible='True', axis='y')  # Ö»ÏÔÊ¾Íø¸ñºáÏß

        plt.ylim(Y_axis[0], Y_axis[1])  # ½öÉèÖÃyÖá×ø±ê·¶Î§
        plt.show()

    def NS_3sub_Graph(Dataset, DataDict, X_axis, Y_axis, ylabel, title='Result Anaylize'):
        matplotlib.rc("font", family='Microsoft YaHei')
        # LineColor = ['#990000', '#4F596D', '#FFFF00', '#70ad47', '#ed7d31', '#CC6633']  # the firset is ours
        LineColor = ['#FF0000', '#6495ED', '#FFA500', '#808000', '#3CB371', '#ed7d31']  # the firset is ours
        LineMarker = ['o', '*', '^', 'x', 'p']
        DataList = list(DataDict.values())
        LabelList = list(DataDict.keys())
        X_axis_len = range(5)

        flg = plt.figure(figsize=(9, 4))
        axes = flg.subplots(nrows=1, ncols=3)
        plt.subplots_adjust(wspace=0.28,
                            hspace=0.2)  # wspace ×ÓÍ¼ºáÏò¼ä¾à£¬ hspace ´ú±í×ÓÍ¼¼äµÄ×ÝÏò¾àÀë£¬left ´ú±íÎ»ÓÚÍ¼Ïñ²»Í¬Î»ÖÃ

        for i in range(len(axes)):
            for j in range(len(DataDict)):
                x_value = DataList[j][i * 5:(i + 1) * 5]
                axes[i].plot(X_axis_len, x_value, color=LineColor[j], linestyle='-', alpha=0.5, linewidth=1.5,
                             marker=LineMarker[j], label=LabelList[j])
                axes[i].set_title(str(Dataset[i]))
                axes[i].set_xticks(X_axis_len, X_axis, rotation=45)
                axes[i].set_label(ylabel)
                axes[i].grid(visible='True', axis='y')  # Ö»ÏÔÊ¾Íø¸ñºáÏß
                axes[i].set_ylim(Y_axis[i][0], Y_axis[i][1])  # ½öÉèÖÃyÖá×ø±ê·¶Î§

        flg.text(0.5, 0, 'K', ha='center', fontsize=16)
        flg.text(0.05, 0.5, ylabel, fontsize=16, va='center', rotation='vertical')

        lines, labels = flg.axes[-1].get_legend_handles_labels()
        # flg.legend(lines, labels, loc = 'upper right')
        plt.show()

    def NS_6sub_Graph(Dataset, DataDict_P, DataDict_NDCG, X_axis, Y_axis_P, Y_axis_NDCG, ylabel,
                      title='Result Anaylize'):
        matplotlib.rc("font", family='Microsoft YaHei')
        LineColor = ['#FF0000', '#6495ED', '#FFA500', '#808000', '#3CB371', '#ed7d31']
        LineMarker = ['o', '*', '^', 'x', 'p']
        DataList_P = list(DataDict_P.values())
        DataList_NDCG = list(DataDict_NDCG.values())
        LabelList = list(DataDict_P.keys())
        X_axis_len = range(5)

        flg = plt.figure(figsize=(10, 4))
        axes = flg.subplots(nrows=2, ncols=3)
        plt.subplots_adjust(wspace=0.28,
                            hspace=0.2)  # wspace ×ÓÍ¼ºáÏò¼ä¾à£¬ hspace ´ú±í×ÓÍ¼¼äµÄ×ÝÏò¾àÀë£¬left ´ú±íÎ»ÓÚÍ¼Ïñ²»Í¬Î»ÖÃ

        for i in range(axes.shape[0]):
            if i == 0:
                for ii in range(axes.shape[1]):
                    for j in range(len(DataDict_P)):
                        x_value = DataList_P[j][ii * 5:(ii + 1) * 5]
                        axes[i][ii].plot(X_axis_len, x_value, color=LineColor[j], linestyle='-',
                                         alpha=0.5, linewidth=1.5, marker=LineMarker[j], label=LabelList[j])
                        axes[i][ii].set_title(str(Dataset[ii]))
                        axes[i][ii].set_xticks(X_axis_len, X_axis, rotation=0)
                        axes[i][ii].set_label(ylabel)
                        axes[i][ii].grid(visible='True', axis='y')  # Ö»ÏÔÊ¾Íø¸ñºáÏß
                        axes[i][ii].set_ylim(Y_axis_P[ii][0], Y_axis_P[ii][1])  # ½öÉèÖÃyÖá×ø±ê·¶Î§
            else:
                for ii in range(axes.shape[1]):
                    for j in range(len(DataDict_NDCG)):
                        x_value = DataList_NDCG[j][ii * 5:(ii + 1) * 5]
                        axes[i][ii].plot(X_axis_len, x_value, color=LineColor[j], linestyle='-',
                                         alpha=0.5, linewidth=1.5, marker=LineMarker[j], label=LabelList[j])
                        # axes[i][ii].set_title(str(Dataset[ii]))
                        axes[i][ii].set_xticks(X_axis_len, X_axis, rotation=0)
                        axes[i][ii].set_label(ylabel)
                        axes[i][ii].grid(visible='True', axis='y')  # Ö»ÏÔÊ¾Íø¸ñºáÏß
                        axes[i][ii].set_ylim(Y_axis_NDCG[ii][0], Y_axis_NDCG[ii][1])  # ½öÉèÖÃyÖá×ø±ê·¶Î§

        flg.text(0.5, 0, 'K1', ha='center', fontsize=16)
        flg.text(0.05, 0.7, "P", fontsize=16, va='center', rotation='vertical')
        flg.text(0.05, 0.3, "NDCG", fontsize=16, va='center', rotation='vertical')

        lines, labels = flg.axes[-1].get_legend_handles_labels()
        labels = ["N=" + i for i in labels]
        flg.legend(lines, labels, loc='upper right')
        plt.show()


