import torch

import torch.nn as nn

m = nn.BatchNorm2d(10,affine=True) #权重w和偏重将被使用
nn.BatchNorm1d()
input = torch.zeros(1,10,10,1)
for i in range(10):
    for j in range(10):
        input[0][i][j][0]=j


print(input)
output = m(input)
print(output)