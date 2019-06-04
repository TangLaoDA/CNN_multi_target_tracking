import CdataSet

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision#数据
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F#激活函数都在这里

EPOCH = 100000
BATCH_SIZE = 100
LR = 0.001

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
            nn.BatchNorm2d(16, affine=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),  # 卷积出来的图片大小不变64  池化后32  128 64
            nn.ReLU6(),
            nn.BatchNorm2d(32, affine=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 卷积出来的图片大小不变32  池化后16 64  32
            nn.ReLU6(),
            nn.BatchNorm2d(64, affine=True),
            nn.MaxPool2d(kernel_size=2,padding=0)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # 卷积出来的图片大小不变16  池化后8 32 16
            nn.ReLU6(),
            nn.BatchNorm2d(128, affine=True),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )
        self.fullc = nn.Linear(16 * 16 * 128, 128)
        self.out = nn.Linear(128,10)
        self.bn=nn.BatchNorm1d(128)
    def forward(self,x):
        fc_hide = (self.conv1(x))
        fc_hide = self.conv2(fc_hide)
        fc_hide = self.conv3(fc_hide)
        fc_hide = self.conv4(fc_hide)
        # print(fc2.size())
        fc2 = fc_hide.view(fc_hide.size(0),-1)#view()相当于reshape()
        # print(fc2.size())
        output = F.relu6(self.fullc(fc2))
        output = F.softmax(self.out(self.bn(output)),dim=1)
        return output

#求解one hot 坐标

if __name__ == '__main__':
    for i in range(1,37):
        my_str = "C.pkl_" + str(i) + "_256_3T"
        cnn = CNN()
        cnn = cnn.cuda()

        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

        loss_func = nn.MSELoss()
        loss_func = loss_func.cuda()

        train_data = CdataSet.MyDataset(dir="MulTarg", transform=torchvision.transforms.ToTensor(), index=i)
        train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        plt.ion()
        for epoch in range(EPOCH):
            for i, (x, y) in enumerate(train_loader):
                # print(x.size())
                batch_x = Variable(x).float()
                batch_x = batch_x.cuda()
                batch_y = Variable(y).float()
                batch_y = batch_y.cuda()
                # 标签转onehot
                # y = torch.unsqueeze(y, dim=1)
                # batch_y = Variable(y).long()
                # batch_y = (torch.zeros(BATCH_SIZE, 2).scatter_(1, batch_y, 1)).float().cuda()
                # batch_x=batch_x.permute(0,2,3,1)
                # 输入训练数据
                output = cnn(batch_x)

                # 计算误差
                loss = loss_func(output, batch_y)
                # 清空上一次的梯度
                optimizer.zero_grad()
                # 误差反向传播
                loss.backward()
                # 优化器更新参数
                optimizer.step()
                if i % 10 == 0:
                    # print(output[0])
                    # print("********",x[0].size())
                    # # x[0] = x[0].permute((1,2,0))
                    # img=x[0].numpy()
                    # img=img.reshape(3,-1)
                    # img.transpose((1,0))
                    # img = img.reshape(300,300,3)
                    # print("!!!!!!!!!  ",img.shape)
                    # plt.imshow(img)
                    # plt.pause(0.1)
                    # print(batch_y[0])
                    print(loss.item())

                    # for i in range(len(output)):
                    #     if output[i]>0.5:
                    #         output[i]=1.0
                    #     else:
                    #         output[i]=0.0

                    # print(epoch,"+",i,"   准确率： ",torch.mean((torch.tensor((Variable(y).float().cuda() ==output ))).float()))

                if loss.item() <= 0.01:
                    torch.save(cnn, my_str)
                    break
            if loss.item() <= 0.01:
                break





