from PIL import Image
import torch
import torch.utils.data as data
import os
import torchvision
import matplotlib.pyplot as plt
import time
import numpy as np

pic_size=256
class MyDataset(data.Dataset):
    #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,dir, transform=None, target_transform=None):
        imgs = []
        #初始化一些需要传入的参数
        for filename in os.listdir(dir):
            sdir=filename
            x = filename.split(".")
            slabel=int(x[13])
            if slabel == 0:
                slabel=[1,0]
            else:
                slabel = [0, 1]
            imgs.append((sdir, np.array(slabel)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.dir=dir

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(self.dir + r"/"+fn).convert('L')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        img_rgb=Image.open(self.dir + r"/"+fn).convert('RGB')
        img_rgb = np.array(img_rgb,dtype=np.uint8)
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
        img_rgb = torch.from_numpy(img_rgb).byte()
        return img_rgb,img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs) #根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器

if __name__ == '__main__':
    plt.ion()
    train_data = MyDataset(dir="train_sample", transform=torchvision.transforms.ToTensor())
    train_loader = data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)

    for i, (img, x, y) in enumerate(train_loader):
        print(img.size())
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
