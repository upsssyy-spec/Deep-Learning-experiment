import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01#步长
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

#加载数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='../data',
                               train=True,
                               download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
examples = enumerate(train_loader)
batch_idx,(example_data,example_targets) = next(examples)
print(example_targets)
print(example_data.shape)

plt.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
    plt.title('目标数据标签；{}'.format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

