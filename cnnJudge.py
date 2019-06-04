import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision#数据
import torch.utils.data as data

EPOCH = 2
BATCH_SIZE = 50
LR = 0.001

train_data = torchvision.datasets.MNIST(
    root="mnist_data/",#保存位置
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
test_data = torchvision.datasets.MNIST(
    root="mnist_data/",
    train=False,
    transform=torchvision.transforms.ToTensor()
)

print("train_data:",train_data.train_data.size())
print("train_labels:",train_data.train_labels.size())
print("test_data:",test_data.test_data.size())

train_loader = data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),#卷积出来的图片大小不变
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#2*2采样(16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),  # 卷积出来的图片大小不变
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 2*2采样(32,7,7)
        )
        self.out = nn.Linear(32*7*7,10)
    def forward(self,x):
        fc1 = self.conv1(x)
        fc2 = self.conv2(fc1)
        # print(fc2.size())
        fc2 = fc2.view(fc2.size(0),-1)#view()相当于reshape()
        # print(fc2.size())
        output = self.out(fc2)
        return output

cnn = CNN()
print(cnn)

#opt
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)

#loss_fun
loss_func = nn.CrossEntropyLoss()

#train
for epoch in range(EPOCH):
    for i,(x,y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        print(batch_y.size())
        print(batch_y[0])
        #输入训练数据
        output = cnn(batch_x)
        print(output.size())
        print(output[0][0])
        #计算误差
        loss = loss_func(output,batch_y)
        # loss = nn.CrossEntropyLoss(output,batch_y)
        #清空上一次的梯度
        optimizer.zero_grad()
        #误差反向传播
        loss.backward()
        #优化器更新参数
        optimizer.step()
        if i % 100 == 0:
            print(loss.item())



