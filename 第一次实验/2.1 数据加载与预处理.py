# 代码2-1 
# 使用torchtext.utils.download_from_url类下载数据文件
'''
import torchtext

url = 'https://mirrors.cloud.tencent.com/pypi/packages/51/59/d190ffe6fac2f5c7a301bd9ffd5f69b9a1925e08dbfecd6ee1e8b816fedb/torchtext-0.9.0-cp38-cp38-win_amd64.whl'
torchtext.utils.download_from_url(url, '../data/torchtext.whl')



# 代码2-2
from torchtext.utils import unicode_csv_reader
import io

data_path = '../data/city_economy.csv'
with io.open(data_path, encoding='utf8') as f:
    reader = unicode_csv_reader(f)



# 代码2-3 
# 使用torchtext.utils.extract_archive类读取数据文件
from_path = '../data/test_data.zip'
to_path = '../data/'
torchtext.utils.download_from_url(url, from_path)
torchtext.utils.extract_archive(from_path, to_path)



# 代码2-4 
# 使用torchvision.transforms.Compose类组合图像多种变换处理
from torchvision import transforms

transforms.Compose([transforms.CenterCrop(10), transforms.ToTensor()])



# 代码2-5 
from torchvision.transforms import FiveCrop
# 使用torchvision.transforms.FiveCrop类裁剪图像
transform = transforms.Compose([
        FiveCrop(size),  # size是读取的PIL图像
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))  # 返回4维张量
        ])
input, target = batch  # input是维张量, target是2维张量
bs, ncrops, c, h, w = input.size()
result = model(input.view(-1, c, h, w))  # 改变数据维度
result_avg = result.view(bs, ncrops, -1).mean(1)  # 求均值



# 代码2-6 
# 使用句子计数器
from torchtext.data.functional import sentencepiece_numericalizer

sp_id_generator = sentencepiece_numericalizer(sp_model)
list_a = ['sentencepiece encode as pieces', 'examples to   try!']
list(sp_id_generator(list_a))



# 代码2-7 
# 使用句子分词器对文本句子成分进行标记
from torchtext.data.functional import sentencepiece_tokenizer

sp_tokens_generator = sentencepiece_tokenizer(sp_model)
list_a = ['sentencepiece encode as pieces', 'examples to   try!']
list(sp_tokens_generator(list_a))



# 代码2-8 
# 使用句子分词器对文本句子做分词处理
from torchtext.data.functional import simple_space_split

list_a = ['Sentencepiece encode as pieces', 'example to try!']
list(simple_space_split(list_a))



# 代码2-9 
# 包含路径与标签的列表
# 得到一个包含路径与标签的列表
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F

def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])

    return data



# 代码2-10 
# 参数传入
path1 = '../data/training_data/cats/cat.%d.jpg'
data1 = init_process(path1, [0, 500])



# 代码2-11 
# 判断猫狗
def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first: last]
    if name == 'dog':
        return 1
    else:
        return 0
    
    
    
# 代码2-12 
# 经过函数处理后的四个列表
path1 = '../data/training_data/cats/cat.%d.jpg'
data1 = init_process(path1, [0, 500])
path2 = '../data/training_data/dogs/dog.%d.jpg'
data2 = init_process(path2, [0, 500])
path3 = '../data/testing_data/cats/cat.%d.jpg'
data3 = init_process(path3, [1000, 1200])
path4 = '../data/testing_data/dogs/dog.%d.jpg'
data4 = init_process(path4, [1000, 1200])



# 代码2-13 
# 利用Image处理图片
def Myloader(path):
    return Image.open(path).convert('RGB')



# 代码2-14 
# 重写Dataset类
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
    
    
    
# 代码2-15 
# transform
transform = transforms.Compose([
          transforms.RandomHorizontalFlip(p=0.3),
          transforms.RandomVerticalFlip(p=0.3),
          transforms.Resize((256, 256)),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])



'''
# 代码2-16 
# 猫狗分类的数据预处理
import torch
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
    path1 = 'data/training_data/cats/cat.%d.jpg'
    data1 = init_process(path1, [0, 500])
    path2 = 'data/training_data/dogs/dog.%d.jpg'
    data2 = init_process(path2, [0, 500])
    path3 = 'data/testing_data/cats/cat.%d.jpg'
    data3 = init_process(path3, [1000, 1200])
    path4 = 'data/testing_data/dogs/dog.%d.jpg'
    data4 = init_process(path4, [1000, 1200])
    # 1300个训练
    train_data = data1 + data2 + data3[0: 150] + data4[0: 150]

    train = MyDataset(train_data, transform=transform, loder=Myloader)
    # 100个测试
    test_data = data3[150: 200] + data4[150: 200]
    test = MyDataset(test_data, transform=transform, loder=Myloader)

    train_data = DataLoader(dataset=train, batch_size=10, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0)

    return train_data, test_data


if __name__ == '__main__':
    train_loader, test_loader = load_data()
