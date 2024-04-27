import pandas as pd
from Model import params




NS_method = params.methods['MPS'][2]
dataset = params.dataset[0]
url_train = '../../../Data/'+dataset+'/initial/'+dataset+'.base'
train_data = pd.read_csv(url_train, sep='\t', header=None, index_col=False)

url = '../../../Data/'+dataset+'/intermediate/' + NS_method
K_count = [50, 100, 150, 200, 250, 300, 350, 400]
# K_count = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
count_100k = [100,273,488,731, 1035, 1422, 1862, 2366,]
# count_ls = [1869,3711,5666,7520, 9134, 10663, 12147, 13423, 14636, 15682, 16750, 17926, 19256, 20609]
# couunt_mod = [30, 124, 270, 519, 921, 1588, 2819, 4970, 8177, 10644]
for i, k in enumerate(K_count):
    index = train_data[0].sample(n = count_100k[i]).index
    random_trData = train_data.drop(index).loc[:,:2]
    name = '/' + params.methods['FPS'][-1] + '_mPS_k' + str(k) + '2.base'
    random_trData.to_csv(url+name, header=None, sep='\t', index=False)
    print(k, 'is end.')

print("It is over...")





