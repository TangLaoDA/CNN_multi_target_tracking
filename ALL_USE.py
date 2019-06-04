# import JdataSet
import rgb_use_dataset

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision#数据
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F#激活函数都在这里
import time
import cv2

EPOCH = 1
BATCH_SIZE = 2
LR = 0.001



train_data = rgb_use_dataset.MyDataset(dir="MulTtest", transform=torchvision.transforms.ToTensor())
train_loader = data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

# from J import CNN
# cnn = torch.load("J.pkl_1_256")
# cnn = cnn.cuda()

from C import CNN
cnn1 = torch.load('C.pkl_1_256_3T').cuda()
cnn2 = torch.load('C.pkl_2_256_3T').cuda()
cnn3 = torch.load('C.pkl_3_256_3T').cuda()
cnn4 = torch.load('C.pkl_4_256_3T').cuda()
cnn5 = torch.load('C.pkl_5_256_3T').cuda()
cnn6 = torch.load('C.pkl_6_256_3T').cuda()
cnn7 = torch.load('C.pkl_7_256_3T').cuda()
cnn8 = torch.load('C.pkl_8_256_3T').cuda()
cnn9 = torch.load('C.pkl_9_256_3T').cuda()
cnn10 = torch.load('C.pkl_10_256_3T').cuda()
cnn11 = torch.load('C.pkl_11_256_3T').cuda()
cnn12 = torch.load('C.pkl_12_256_3T').cuda()

cnn13 = torch.load('C.pkl_13_256_3T').cuda()
cnn14 = torch.load('C.pkl_14_256_3T').cuda()
cnn15 = torch.load('C.pkl_15_256_3T').cuda()
cnn16 = torch.load('C.pkl_16_256_3T').cuda()
cnn17 = torch.load('C.pkl_17_256_3T').cuda()
cnn18 = torch.load('C.pkl_18_256_3T').cuda()
cnn19 = torch.load('C.pkl_19_256_3T').cuda()
cnn20 = torch.load('C.pkl_20_256_3T').cuda()
cnn21 = torch.load('C.pkl_21_256_3T').cuda()
cnn22 = torch.load('C.pkl_22_256_3T').cuda()
cnn23 = torch.load('C.pkl_23_256_3T').cuda()
cnn24 = torch.load('C.pkl_24_256_3T').cuda()

cnn25 = torch.load('C.pkl_25_256_3T').cuda()
cnn26 = torch.load('C.pkl_26_256_3T').cuda()
cnn27 = torch.load('C.pkl_27_256_3T').cuda()
cnn28 = torch.load('C.pkl_28_256_3T').cuda()
cnn29 = torch.load('C.pkl_29_256_3T').cuda()
cnn30 = torch.load('C.pkl_30_256_3T').cuda()
cnn31 = torch.load('C.pkl_31_256_3T').cuda()
cnn32 = torch.load('C.pkl_32_256_3T').cuda()
cnn33 = torch.load('C.pkl_33_256_3T').cuda()
cnn34 = torch.load('C.pkl_34_256_3T').cuda()
cnn35 = torch.load('C.pkl_35_256_3T').cuda()
cnn36 = torch.load('C.pkl_36_256_3T').cuda()



loss_func = nn.MSELoss()
loss_func = loss_func.cuda()

save_count=0

for epoch in range(1):
    for i, (img,x, y) in enumerate(train_loader):
        batch_x = Variable(x).float()
        batch_x = batch_x.cuda()
        batch_y = Variable(y).float()
        batch_y = batch_y.cuda()


        # output = cnn(batch_x)
        #第一组坐标
        c1 = cnn1(batch_x)
        c2 = cnn2(batch_x)
        c3 = cnn3(batch_x)
        c4 = cnn4(batch_x)
        c5 = cnn5(batch_x)
        c6 = cnn6(batch_x)
        c7 = cnn7(batch_x)
        c8 = cnn8(batch_x)
        c9 = cnn9(batch_x)
        c10 = cnn10(batch_x)
        c11 = cnn11(batch_x)
        c12 = cnn12(batch_x)
        # 第二组坐标
        c13 = cnn13(batch_x)
        c14 = cnn14(batch_x)
        c15 = cnn15(batch_x)
        c16 = cnn16(batch_x)
        c17 = cnn17(batch_x)
        c18 = cnn18(batch_x)
        c19 = cnn19(batch_x)
        c20 = cnn20(batch_x)
        c21 = cnn21(batch_x)
        c22 = cnn22(batch_x)
        c23 = cnn23(batch_x)
        c24 = cnn24(batch_x)
        # 第三组坐标
        c25 = cnn25(batch_x)
        c26 = cnn26(batch_x)
        c27 = cnn27(batch_x)
        c28 = cnn28(batch_x)
        c29 = cnn29(batch_x)
        c30 = cnn30(batch_x)
        c31 = cnn31(batch_x)
        c32 = cnn32(batch_x)
        c33 = cnn33(batch_x)
        c34 = cnn34(batch_x)
        c35 = cnn35(batch_x)
        c36 = cnn36(batch_x)

        for i in range(BATCH_SIZE):
            if True:
                print("*****检测到目标*****************************")
                print("*****计算目标位置*********************")
                #第一组坐标
                x11=(torch.argmax(c1[i], dim=0))*100+(torch.argmax(c2[i], dim=0))*10+(torch.argmax(c3[i], dim=0))
                x11=x11.cpu()
                y11 = (torch.argmax(c4[i], dim=0)) * 100 + (torch.argmax(c5[i], dim=0)) * 10 + (
                    torch.argmax(c6[i], dim=0))
                y11=y11.cpu()
                x12 = (torch.argmax(c7[i], dim=0)) * 100 + (torch.argmax(c8[i], dim=0)) * 10 + (
                    torch.argmax(c9[i], dim=0))
                x12=x12.cpu()
                y12 = (torch.argmax(c10[i], dim=0)) * 100 + (torch.argmax(c11[i], dim=0)) * 10 + (
                    torch.argmax(c12[i], dim=0))
                y12=y12.cpu()
                #第二组坐标
                x21 = (torch.argmax(c13[i], dim=0)) * 100 + (torch.argmax(c14[i], dim=0)) * 10 + (
                    torch.argmax(c15[i], dim=0))
                x21 = x21.cpu()
                y21 = (torch.argmax(c16[i], dim=0)) * 100 + (torch.argmax(c17[i], dim=0)) * 10 + (
                    torch.argmax(c18[i], dim=0))
                y21 = y21.cpu()
                x22 = (torch.argmax(c19[i], dim=0)) * 100 + (torch.argmax(c20[i], dim=0)) * 10 + (
                    torch.argmax(c21[i], dim=0))
                x22 = x22.cpu()
                y22 = (torch.argmax(c22[i], dim=0)) * 100 + (torch.argmax(c23[i], dim=0)) * 10 + (
                    torch.argmax(c24[i], dim=0))
                y22 = y22.cpu()
                # 第三组坐标
                x31 = (torch.argmax(c25[i], dim=0)) * 100 + (torch.argmax(c26[i], dim=0)) * 10 + (
                    torch.argmax(c27[i], dim=0))
                x31 = x31.cpu()
                y31 = (torch.argmax(c28[i], dim=0)) * 100 + (torch.argmax(c29[i], dim=0)) * 10 + (
                    torch.argmax(c30[i], dim=0))
                y31 = y31.cpu()
                x32 = (torch.argmax(c31[i], dim=0)) * 100 + (torch.argmax(c32[i], dim=0)) * 10 + (
                    torch.argmax(c33[i], dim=0))
                x32 = x32.cpu()
                y32 = (torch.argmax(c34[i], dim=0)) * 100 + (torch.argmax(c35[i], dim=0)) * 10 + (
                    torch.argmax(c36[i], dim=0))
                y32 = y32.cpu()


                if i==0:
                    n_array = img.numpy()


                m_array = n_array[i]
                print(n_array.shape)

                m_array = np.array(m_array, dtype=np.uint8)

                print(x11,"  ",y11,"  ",x12," ",y12)
                img = cv2.rectangle(m_array, (x11, y11), (x12, y12), color=0xff00)
                img = cv2.rectangle(img, (x21, y21), (x22, y22), color=0xff00)
                img = cv2.rectangle(img, (x31, y31), (x32, y32), color=0xff00)
                cv2.imshow("yello", img)
                #cv2.imwrite(r"multiSave\{0}.png".format(save_count),img)
                save_count+=1
                cv2.waitKey(0)

                cv2.destroyAllWindows()

            else:
                print("*****未检测到目标，程序结束***********")
