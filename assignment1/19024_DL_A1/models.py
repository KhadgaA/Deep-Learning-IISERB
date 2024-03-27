import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_out(W,K,P,S):
    return ((W-K + 2*P)//S)+1
def pool_out(I,F,S):
    return ((I-F)//S )+ 1
def layer_out(lyr:list,inp:dict): #['conv','pool','conv','pool'], inp ={'W','K','P','SC','IP','FP','SP'}
    '''Function to find the out size after a series of Conv and Pool operations'''
    WC, KC, PC, SC,FP, SP = inp['WC'], inp['KC'], inp['PC'], inp['SC'],inp['FP'],inp['SP']
    # print(WC, KC, PC, SC, FP, SP)
    for lr in lyr:
        if lr == 'conv':
            WC = conv_out(WC,KC,PC,SC)
        elif lr =='pool':
            WC = pool_out(I = WC,F=FP,S=SP)
    return WC
class model1(nn.Module):
    # Conv-Pool-Conv-Pool-Conv-Pool-FC
    def __init__(self,image_size = 32,out = 10,channels = 3,pool_size = 3,stride = 2):
        super(model1, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.out = out
        self.conv1 = nn.Conv2d(3,32,3,1)# [(W−K+2P)/S]+1.
        self.conv2 = nn.Conv2d(32, 64, 3, 1) #[32 - 3 + 0 /1] = 29
        self.conv3 = nn.Conv2d(64, 128, 3, 1)# [29 - 3 + 0 / 1] = 26
        self.pool = nn.MaxPool2d(pool_size,stride=stride) #[26 - 3]/2 +1   # [(I - F) / S] + 1 x D ; I input, F kernel size, D depth/dimension, S stride
        fc_in = layer_out(['conv','pool','conv','pool','conv','pool'],
                          {'WC':32,'KC':3,'PC':0,'SC':1,'FP':pool_size,'SP':stride})
        self.fc = nn.Linear(128*(fc_in**2),10) #  - 3)/2 + 1 x 128
    def forward(self, x):
        x = self.conv1(x) # 3x32x32 -> [32 - 3 + 0]/1 + 1 = 30
        x = self.pool(x)  # 32x30x30 -> [30 - 3]/2 + 1 xD =  32 x14
        x = self.conv2(x) #
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        # print(x.shape)
        return x


class model2(nn.Module):
    # Conv-Conv-Pool-Conv-Conv-Pool-FC
    def __init__(self,image_size = 32,channels = 3,pool_size = 3,stride = 2):
        super(model2, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.pool = nn.MaxPool2d(pool_size,stride=stride)
        fc_in = layer_out(['conv', 'conv', 'pool', 'conv', 'conv', 'pool'],
                          {'WC': 32, 'KC': 3, 'PC': 0, 'SC': 1, 'FP': pool_size, 'SP': stride})
        self.fc = nn.Linear(256*(fc_in**2),10)
    def forward(self, x):
        x = self.conv1(x) # 3x32x32 ->
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        # print(x.shape)
        return x

class model3(nn.Module):
    # Conv-Pool-Conv-Pool-Conv-Pool-FC-FC
    def __init__(self,image_size = 32,channels = 3,pool_size = 3,stride = 2):
        super(model3, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(pool_size,stride=stride)
        fc1_in = layer_out(['conv','pool','conv','pool','conv','pool'],
                          {'WC': 32, 'KC': 3, 'PC': 0, 'SC': 1, 'FP': pool_size, 'SP': stride})
        self.fc1 = nn.Linear(128*(fc1_in**2),512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.conv1(x) # 3x32x32 ->
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)
        return x
if __name__ =='__main__':

   img = torch.rand(2, 3, 32, 32)
   out = model3(stride=1,pool_size=2)(img)
   print(out.shape)
