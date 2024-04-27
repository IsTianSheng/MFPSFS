import torch
import torch.nn as nn


class discriminator(nn.Module):

    def __init__(self,itemCount):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(itemCount,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,128),
            nn.LeakyReLU(),
            nn.Linear(128,16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self,data):
        result=self.dis( data )
        return result


class discriminator_inital(nn.Module):
    def __init__(self, itemCount, info_shape):
        super(discriminator_inital, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(itemCount + info_shape, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, data, condition):
        data_c = torch.cat((data, condition), 1)
        result = self.dis(data_c)
        return result




# The init
class generator_inital(nn.Module):

    def __init__(self,itemCount,info_shape):
        self.itemCount = itemCount
        super(generator_inital,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(self.itemCount+info_shape, 256),
            nn.ReLU(True),          # inplace=True  Êý¾Ý½øÐÐreluÔËËãºóÊäÈë±ä»¯Óë·ñµÄ¿ª¹Ø
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024, itemCount),
            nn.Sigmoid()
        )

    def forward(self,noise,useInfo):
        G_input = torch.cat([noise, useInfo], 1)
        result=self.gen(G_input)
        return result



class generator_no_userInfo(nn.Module):

    def __init__(self,itemCount):
        self.itemCount = itemCount
        super(generator_no_userInfo,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(self.itemCount, 256),
            nn.ReLU(True),          # inplace=True  Êý¾Ý½øÐÐreluÔËËãºóÊäÈë±ä»¯Óë·ñµÄ¿ª¹Ø
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024, itemCount),
            nn.Sigmoid()
        )

    def forward(self,noise):
        result=self.gen(noise)
        return result



# The changed model--two input
class generator(nn.Module):

    def __init__(self,itemCount,userCount):
        self.itemCount = itemCount
        super(generator,self).__init__()
        self.linear1 = nn.Linear(self.itemCount, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(userCount, 512)
        self.linear4 = nn.Linear(512, 512)

        self.linear5 = nn.Linear(512, 1024)
        self.linear6 = nn.Linear(1024, self.itemCount)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()


    def forward(self,noise,neighbors):

        # G_input = torch.cat([noise, useInfo], 1)
        out1 = self.relu(self.linear1(noise))
        out1 = self.relu(self.linear2(out1))

        out2 = self.relu(self.linear3(neighbors))
        out2 = self.relu(self.linear4(out2))

        # print(type(out2),out1.shape,out2.shape)
        # out = torch.cat([out1, out2], 1)
        out = torch.mul(out1, out2)
        out = self.relu(self.linear5(out))
        out = self.sigmoid(self.linear6(out))
        return out




