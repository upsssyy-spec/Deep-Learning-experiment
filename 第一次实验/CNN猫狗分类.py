# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:55:53 2021

@author: zhang
"""

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F


def Myloader(path):
    return Image.open(path).convert('RGB')


# 得到一个包含路径与标签的列表
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])

    return data


class MyDataset(Dataset):

    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'dog':
        return 1
    else:
        return 0


def load_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    path1 = '../data/training_data/cats/cat.%d.jpg'
    data1 = init_process(path1, [0, 500])
    path2 = '../data/training_data/dogs/dog.%d.jpg'
    data2 = init_process(path2, [0, 500])
    path3 = '../data/testing_data/cats/cat.%d.jpg'
    data3 = init_process(path3, [1000, 1200])
    path4 = '../data/testing_data/dogs/dog.%d.jpg'
    data4 = init_process(path4, [1000, 1200])
    # 1300个训练
    train_data = data1 + data2 + data3[0: 150] + data4[0: 150]

    train = MyDataset(train_data, transform=transform, loder=Myloader)
    # 100个测试
    test_data = data3[150: 200] + data4[150: 200]
    test = MyDataset(test_data, transform=transform, loder=Myloader)

    train_data = DataLoader(dataset=train, 
                            batch_size=10, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, 
                           batch_size=1, shuffle=True, num_workers=0)

    return train_data, test_data


# 三种网络择优选择其中一种即可
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.output = nn.Linear(16 * 64 * 64, 2)  
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        temp = x.view(x.size()[0], -1)
        x = self.dp1(x)
        output = self.output(temp)
        return output, x



class MYVGG(nn.Module):
    def __init__(self, num_classes=2):
        super(MYVGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(256, 256, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(256, 512, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.output = nn.Linear(512, num_classes)
        self.dp1 = nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.pool(F.relu(self.conv8(x)))
        temp = x.view(x.size()[0], -1)
        x = self.dp1(x)
        output = self.output(temp)
        return output, x



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.output = nn.Linear(in_features=512 * 6 * 6,
                                out_features=2)
        self.dp1 = nn.Dropout(p=0.5)
        
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        temp = x.view(x.shape[0], -1)
        x = self.dp1(x)
        output = self.output(temp)
        return output, x


def train():
    train_loader, test_loader = load_data()
    epoch_num = 200
    # GPU计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00004)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(
                    target.long()).to(device)
            optimizer.zero_grad()  # 梯度清0
            output = model(data)[0]  # 前向传播
            loss = criterion(output, target)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    torch.save(model, '../tmp/cnn.pkl')


def test():
    train_loader, test_loader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('../tmp/cnn.pkl')  # load model
    total = 0
    current = 0
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]

        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()

    print('Accuracy: %d %%' % (100 * current / total))


if __name__ == '__main__':
    train()
    test()
