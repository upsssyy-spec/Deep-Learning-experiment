import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from PIL import Image


# 定义创建关于图像路径txt文件的函数
def classes_txt(root, out_path, num_class=None):
    # 列出根目录下所有类别所在文件夹名
    dirs = os.listdir(root)
    # 不指定类别数量就读取所有
    if not num_class:
        num_class = len(dirs)
    # 输出文件路径不存在就新建
    if not os.path.exists(out_path):
        f = open(out_path, 'w')
        f.close()
	# 如果文件中本来就有一部分内容，只需要补充剩余部分
	# 如果文件中数据的类别数比需要的多就跳过
    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                for file in files:
                    f.write(os.path.join(root, dir, file) + '\n')


# 定义读取并变换数据格式的类
class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        # 存储图像的路径
        images = []
        # 图像的类别名，在本例中是数字
        labels = []
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:
                # 只读取前 num_class个类
                if int(line.split('\\')[-2]) >= num_class:
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('\\')[-2]))
        self.images = images
        self.labels = labels
        # 图片需要进行的格式变换，比如ToTensor()等等
        self.transforms = transforms


    def __getitem__(self, index):
        # 用PIL.Image读取图像
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        if self.transforms is not None:
            # 进行格式变换
            image = self.transforms(image)
        return image, label


    def __len__(self):
        return len(self.labels)


# 首先将训练集和测试集文件途径和文件名以txt保存在一个文件夹中，路径自行定义
root = './data'  # 文件的储存位置
classes_txt(root + '/train', root+'/train.txt')
classes_txt(root + '/test', root+'/test.txt')

# 由于数据集图片尺寸不一，因此要进行resize（重设大小）
# 将图片大小重设为 64 * 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])

# 提取训练集和测试集图片的路径生成txt文件
# num_class 选取100种汉字  提出图片和标签
train_set = MyDataset(root + '/train.txt',
                      num_class=100,
                      transforms=transform)
test_set = MyDataset(root + '/test.txt',
                     num_class =100,
                     transforms = transform)
# 放入迭代器中
train_loader = DataLoader(train_set, batch_size=50, shuffle=True)
test_loader = DataLoader(test_set, batch_size=5473, shuffle=True)
# 这里的5473是因为测试集为5973张图片，当进行迭代时取第二批500个图片进行测试
for step, (x,y) in enumerate(test_loader):
    test_x, labels_test = x.to(device), y.to(device)



# 构建网络
class MYNET(nn.Module):
    def __init__(self):
        super(MYNET, self).__init__()
        # 3个参数分别是输入通道数，输出通道数，卷积核大小
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 100)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 查看网络结构
model = MYNET().to(device)
summary(model, (1, 64, 64))



# 编译网络
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 损失函数
loss_func = nn.CrossEntropyLoss()


# 训练模型
EPOCH = 10
for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        picture, labels = x.to(device), y.to(device)
        output = model(picture)
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 性能评估
        if step % 50 == 0:
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = ((pred_y == labels_test).sum().item() /
                        labels_test.size(0))
            # 输出迭代次数、训练误差、测试准确率
            print('迭代次数:', epoch,
                  '| 训练损失:%.4f' % loss.data,
                  '| 测试准确率:', accuracy)

print('完成训练')


# 保存模型
torch.save(model.state_dict(), './tmp/model.pkl')


# 泛化测试
# 测试图像处理
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])

# 加载模型
model = MYNET()
model.load_state_dict(torch.load('./tmp/model.pkl'))
model.eval()

# 输入图像并预测
img = Image.open('./data/test/00008/816.png')
img = transform(img)
img = img.view(1, 1, 64, 64)
output = model(img)
_, prediction = torch.max(output, 1)
prediction = prediction.numpy()[0]










