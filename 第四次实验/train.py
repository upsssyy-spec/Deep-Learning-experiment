import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置图的大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images): # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape((sqrtimg, sqrtimg)))
    return

def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5

def deprocess_img(x):
    return (x + 1.0) / 2.0

# 代码3-36 定义用于采样的ChunkSampler类
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 50000
NUM_VAL = 5000
NOISE_DIM = 96
batch_size = 128

train_set = MNIST('./', train=True, download=False, transform=preprocess_img)
train_data = DataLoader(train_set, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0))
val_set = MNIST('./', train=True, download=False, transform=preprocess_img)
val_data = DataLoader(val_set, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

# 可视化图片效果
train_data_iter = iter(train_data)  # 创建迭代器
batch_imgs, batch_labels = next(train_data_iter)  # 获取下一批数据

# 处理图片
imgs = deprocess_img(batch_imgs.view(batch_size, 784)).numpy().squeeze()
show_images(imgs)

# 代码3-37 定义判别器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class build_dc_classifier(nn.Module):
    def __init__(self):
        super(build_dc_classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

# 代码3-38 定义生成器
class build_dc_generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super(build_dc_generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7) # 调整形状，通道是128，大小是7×7
        x = self.conv(x)
        return x

# 代码3-39 定义损失函数
bce_loss = nn.BCEWithLogitsLoss()
def discriminator_loss(logits_real, logits_fake): # 判别器的损失函数
    size = logits_real.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float().to(device)
    false_labels = Variable(torch.zeros(size, 1)).float().to(device)
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss

def generator_loss(logits_fake): # 生成器的损失函数
    size = logits_fake.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float().to(device)
    loss = bce_loss(logits_fake, true_labels)
    return loss

# 代码3-40 构建生成对抗网络
def train_dc_gan(D_net,
                 G_net,
                 D_optimizer,
                 G_optimizer,
                 discriminator_loss,
                 generator_loss,
                 show_every=250,
                 noise_size=96,
                 num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            real_data = Variable(x).to(device) # 真实数据
            logits_real = D_net(real_data) # 判别网络得分
            # -1~1的分布
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5
            g_fake_seed = Variable(sample_noise).to(device)
            fake_images = G_net(g_fake_seed) # 生成的数据
            logits_fake = D_net(fake_images) # 判别网络得分

            # 判别器的损失函数
            d_total_error = discriminator_loss(logits_real, logits_fake)
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step() # 优化判别网络

            # 生成网络
            g_fake_seed = Variable(sample_noise).to(device)
            fake_images = G_net(g_fake_seed) # 生成的假的数据
            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake) # 生成器的损失函数
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step() # 优化生成网络

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,
                      d_total_error.item(), g_error.item()))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1

# 代码3-41 定义优化器
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer

# 代码3-42 训练网络
D_DC = build_dc_classifier().to(device)
G_DC = build_dc_generator().to(device)
D_DC_optim = get_optimizer(D_DC) # 判别器创建优化器
G_DC_optim = get_optimizer(G_DC) # 生成器创建优化器
train_dc_gan(D_DC, G_DC, D_DC_optim, G_DC_optim,
             discriminator_loss, generator_loss, num_epochs=20)