from PIL import Image
import torch
import torch.utils.data as data
import os
import torchvision
import matplotlib.pyplot as plt
import time
import numpy as np

pic_size=256
def parseFonc(in_filename,index):
    god_like=index-1
    in_filename = in_filename.split(".")
    in_filename.remove(in_filename[0])
    in_filename.remove(in_filename[13])
    in_filename.remove(in_filename[12])
    out_array = []
    out_array_new = []
    tick_num1 = 0
    while tick_num1 < 12:
        derta = 3 - len(in_filename[tick_num1])
        if derta == 0:
            pass
        elif derta == 1:
            in_filename[tick_num1] = "0" + in_filename[tick_num1]
        elif derta == 2:
            in_filename[tick_num1] = "00" + in_filename[tick_num1]
        else:
            print("errror 111")
        tick_num1 += 1

    tick_num2 = 0
    while tick_num2 < 12:
        temp_str = list(in_filename[tick_num2])  # 得到3个字符
        tick_num3 = 0
        while tick_num3 < 3:
            out_array.append(int(temp_str[tick_num3]))
            tick_num3 += 1
        tick_num2 += 1

    # 有12个数字
    # 取第1个数字
    tick_num4 = out_array[god_like]
    if tick_num4 == 0:
        out_array_new.append(1)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 1:
        out_array_new.append(0)  # 0
        out_array_new.append(1)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 2:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(1)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 3:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(1)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 4:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(1)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 5:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(1)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 6:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(1)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 7:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(1)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 8:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(1)  # 8
        out_array_new.append(0)  # 9
    elif tick_num4 == 9:
        out_array_new.append(0)  # 0
        out_array_new.append(0)  # 1
        out_array_new.append(0)  # 2
        out_array_new.append(0)  # 3
        out_array_new.append(0)  # 4
        out_array_new.append(0)  # 5
        out_array_new.append(0)  # 6
        out_array_new.append(0)  # 7
        out_array_new.append(0)  # 8
        out_array_new.append(1)  # 9

    return out_array_new
class MyDataset(data.Dataset):
    #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,dir, transform=None, target_transform=None,index=None):
        imgs = []
        #初始化一些需要传入的参数
        for filename in os.listdir(dir):
            sdir=filename
            slabel=parseFonc(filename,index)
            imgs.append((sdir, np.array(slabel)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.dir=dir

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(self.dir + r"/"+fn).convert('L')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        # img.show()
        # time.sleep(5)
        # if self.transform is not None:
        #     img = self.transform(img)  # 是否进行transform
        img=img.resize((pic_size,pic_size))
        img=np.array(img)
        img=img/255.0-0.5
        img = img.reshape(-1, 1)
        img.transpose((1, 0))
        img = img.reshape(1, pic_size, pic_size)
        img=torch.from_numpy(img)
        label = torch.from_numpy(label)
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs) #根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器

if __name__ == '__main__':
    plt.ion()
    train_data = MyDataset(dir="MulTarg", transform=torchvision.transforms.ToTensor(),index=3)
    train_loader = data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)

    for i, (x, y) in enumerate(train_loader):
        print(y)
    # for i in range(5000):
    #     img = x[0].numpy()
    #     print(x[0].size())
    #     img = img.reshape(1, -1)
    #     img.transpose((1, 0))
    #     img = img.reshape(pic_size, pic_size)
    #     print("!!!!!!!!!  ", img.shape)
    #     plt.text(0.5, 0, "Loss=%.4f" % (y[0][0].cpu()).data.numpy(), fontdict={"size": 20, "color": "red"})
    #     plt.imshow(img)
    #     plt.pause(0.5)
    #     plt.clf()
        # x,_=train_data.__getitem__(i)
        # img = x.numpy()
        # print(img)
        # print(x.size())
        #
        # img = img.reshape(3, -1)
        # img.transpose((1, 0))
        # img = img.reshape(300, 300, 3)
        # print("!!!!!!!!!  ", img.shape)
        # plt.imshow(img)
        # plt.pause(0.1)
