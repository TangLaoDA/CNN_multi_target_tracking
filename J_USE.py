import JdataSet

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision#数据
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F#激活函数都在这里

EPOCH = 1
BATCH_SIZE = 1
LR = 0.001



train_data = JdataSet.MyDataset(dir="train_sample", transform=torchvision.transforms.ToTensor())


train_loader = data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

print("***********",train_loader.__len__())
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),#卷积出来的图片大小不变  128 64   256  128
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),  # 卷积出来的图片大小不变64  池化后32  128 64
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 卷积出来的图片大小不变32  池化后16 64  32
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=2,padding=0)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # 卷积出来的图片大小不变16  池化后8 32 16
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )
        self.out = nn.Linear(16*16*128,2)
    def forward(self,x):
        fc_hide = (self.conv1(x))
        fc_hide = self.conv2(fc_hide)
        fc_hide = self.conv3(fc_hide)
        fc_hide = self.conv4(fc_hide)
        # print(fc2.size())
        fc2 = fc_hide.view(fc_hide.size(0),-1)#view()相当于reshape()
        # print(fc2.size())
        output = self.out(fc2)
        return output

cnn = CNN()
cnn = torch.load('J.pkl_1_256')
cnn=cnn.cuda()

print(cnn)


optimizer =torch.optim.Adam(cnn.parameters(),lr=LR)





loss_func = nn.MSELoss()
loss_func=loss_func.cuda()


plt.ion()
for epoch in range(EPOCH):
    for i,(x,y) in enumerate(train_loader):
        batch_x = Variable(x).float()
        batch_x=batch_x.cuda()
        batch_y = Variable(y).float()
        batch_y=batch_y.cuda()

        output = cnn(batch_x)

        loss = loss_func(output,batch_y)

        if i % 100 == 0:
            print("***************************")
            print(torch.argmax(output,dim=1))
            print(torch.argmax(batch_y,dim=1))
            print(loss.item())
            print("***************************")







