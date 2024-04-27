import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from Utils.Utils import Utils,data

warnings.filterwarnings("ignore")



#
#
#
def Get_NS_RKNN_I(rating_matrix, interact_matrix, K):
    sim_matrix_5 = cosine_similarity(rating_matrix.T)
    sim_matrix_1 = cosine_similarity(interact_matrix.T)

    # calculate every item`s KNN
    K_nearest = []
    length = rating_matrix.shape[1]
    for i in range(length):
        line_5 = list(sim_matrix_5[i].argsort()[::-1])
        line_1 = list(sim_matrix_1[i].argsort()[::-1])
        line_5.remove(i)
        line_1.remove(i)
        neighbour_1 = line_1[:K]
        neighbour_5 = line_5[:K]
        KRN = list(set(neighbour_1) & set(neighbour_5))
        K_nearest.append(KRN)

    # 1
    # Combine the KNN_items of the set of items which user perchase
    # Noted that
    K_nearest = np.array(K_nearest)
    RKNN = []
    for i in range(rating_matrix.shape[0]):
        indexlist = list(np.argwhere(interact_matrix[i] == 1)[:, 0])
        itemlist = list(set(sum(K_nearest[indexlist], [])))
        itemlist = [i for i in itemlist if i not in indexlist]
        RKNN.append(itemlist)

    return RKNN


# get items which user`s Farest neighbors parchase
def Get_NS_RKFN_U(rating_matrix, interact_matrix, K):
    sim_matrix_5 = cosine_similarity(rating_matrix)
    sim_matrix_1 = cosine_similarity(interact_matrix)

    # calculate every item`s KFN
    K_farest = []
    length = rating_matrix.shape[0]
    for i in range(length):
        line_5 = list(sim_matrix_5[i].argsort())
        line_1 = list(sim_matrix_1[i].argsort())
        line_5.remove(i)
        line_1.remove(i)
        neighbour_1 = line_1[:K]
        neighbour_5 = line_5[:K]
        KRN = list(set(neighbour_1) & set(neighbour_5))
        K_farest.append(KRN)

    # 1
    # Combine the KNN_items of the set of items which user perchase
    K_farest = np.array(K_farest)
    RKFN = []
    for i in range(rating_matrix.shape[0]):  # one user
        neighbor_items = []
        indexlist = list(np.argwhere(interact_matrix[i] == 1)[:, 0])
        for uid in K_farest[i]:
            neighbor_items_indexlist = list(np.argwhere(interact_matrix[uid] == 1)[:, 0])
            neighbor_items.append(neighbor_items_indexlist)
        itemlist = list(set(sum(neighbor_items, [])))
        itemlist = [i_ for i_ in itemlist if i_ not in indexlist]
        RKFN.append(itemlist)

    return RKFN


def get_NS_u(RKNN_user, rating_matrix):
    NS_ = np.zeros_like(rating_matrix)
    for i in range(rating_matrix.shape[0]):
        ns_list = list(set(RKNN_user[i]))
        NS_[i][ns_list] = 1
    return NS_


def get_NS(RKNN_user, RKNN_item, rating_matrix):
    NS_ = np.zeros_like(rating_matrix)
    for i in range(rating_matrix.shape[0]):
        ns_list = list(set(RKNN_user[i] + RKNN_item[i]))
        NS_[i][ns_list] = 1
    return NS_


# get each user`s k_reciprocal neighbors
def Get_KRN(sim_matrix, k_nearest_num, k_reciprocal_num):
    # sim_matrix = cosine_similarity(matrix)
    K_nearest = []
    for userId, line in enumerate(sim_matrix):
        line = line.argsort()[::-1]
        neighbour = line[1:k_nearest_num + 1]
        K_nearest.append(neighbour)
    K_Reciprocal = []
    for userId, kneighbors in enumerate(K_nearest):
        k_reciprocal_list = np.argwhere(np.array(K_nearest) == userId)[:, 0]
        k_reciprocal_list = list(k_reciprocal_list)
        K_Reciprocal.append(k_reciprocal_list)
    return K_Reciprocal


# improve the method
def select_NS_Neighbor(rating_matrix, interact_matrix, k_nearest_num, k_reciprocal_num):
    sim_matrix_5 = cosine_similarity(rating_matrix)
    sim_matrix_1 = cosine_similarity(interact_matrix)
    KRNN_1 = Get_KRN(sim_matrix_1, k_nearest_num, k_reciprocal_num)
    KRNN_5 = Get_KRN(sim_matrix_5, k_nearest_num, k_reciprocal_num)
    length = rating_matrix.shape[0]
    KRNN = np.zeros((length, length))
    index = [i for i in range(length)]
    for i in index:
        # combine_KRN = list(set(KRNN_1[i]) | set(KRNN_5[i]))
        combine_KRN = list(set(KRNN_1[i]) & set(KRNN_5[i]))
        KRNN[i][combine_KRN] = 1

    return KRNN


if __name__ == '__main__':

    TopN = [5, 10, 20]
    epochs = 1000
    pro_ZR = 50
    pro_PM = 50
    alpha = 0.1

    trainSet, train_use, train_item, interact_matrix, rating_matrix = data.loadTrainingData_modify(
        "../../Data/ml100k/ml100k.base", "\t")
    testSet, test_use, test_item = data.loadTestData("../../Data/ml100k/ml100k.test", "\t")
    userCount = max(train_use, test_use)
    itemCount = max(train_item, test_item)
    userList_test = list(testSet.keys())
    trainVector, trainMaskVector, batchCount = data.to_Vectors(trainSet, userCount, \
                                                               itemCount, userList_test, "userBased")

    K = 5
    RKNN_u = Get_NS_RKFN_U(rating_matrix, interact_matrix, K)
    RKNN_i = Get_NS_RKNN_I(rating_matrix, interact_matrix, K)
    NS_i = get_NS_u(RKNN_i, rating_matrix)
    NS_u = get_NS_u(RKNN_u, rating_matrix)
    NS = get_NS(RKNN_u, RKNN_i, rating_matrix)

    data.saveNS(NS,'../../Data/ml100k/u1_ns.Data')







