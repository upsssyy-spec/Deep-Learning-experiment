# %%
# 导入程序所需要的包

# PyTorch需要的包
import torch
import torch.utils.data as DataSet
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 计算需要的包
import string
import numpy as np
import time

# %%
# 读入并展示数据
f = open('./dataset/poems_clean.txt', "r", encoding='utf-8')
poems = []
for line in f.readlines():
    title, poem = line.split(':')
    poem = poem.replace(' ', '')
    poem = poem.replace('\n', '')
    if len(poem) > 0:
        poems.append(list(poem))

print(poems[0][:])
# %%
# 创建字符编码字典
word2idx = {}
i = 1
for poem in poems:
    for word in poem:
        if word2idx.get(word) == None:
            word2idx[word] = i
            i += 1

print(f"字典大小: {len(word2idx)}")
# %%
# 对诗歌进行编码，从原始数据到矩阵
poems_digit = []
for poem in poems:
    poem_digit = []
    for word in poem:
        poem_digit.append(word2idx[word])
    poems_digit.append(poem_digit)

print("原始诗歌")
print(poems[3829])
print("\n编码后的结果")
print(poems_digit[3829][:])
# %%
# 拆分X、Y变量并处理长短不一问题
# 设置诗歌最大长度为50个字符
maxlen = 50
X = []
Y = []
for poem_digit in poems_digit:
    # 将每首诗歌的最后一个字符作为Y
    Y.append(poem_digit[-1])
    # 将最后一个字符之前的部分作为X，并补齐字符
    x = poem_digit[:-1] + [0] * (maxlen - len(poem_digit))
    X.append(x)

print("原始诗歌")
print(poems[3829])
print("变量X")
print(X[3829])
print("变量Y")
print(Y[3829])
# %%
# 划分训练集和测试集
# 将所有数据的顺序打乱重排
idx = np.random.permutation(range(len(X)))
X = [X[i] for i in idx]
Y = [Y[i] for i in idx]

# 切分出1/5的数据放入校验集
validX = X[:len(X) // 5]
trainX = X[len(X) // 5:]
validY = Y[:len(Y) // 5]
trainY = Y[len(Y) // 5:]
# %%
# 形成训练集
batch_size = 64
train_ds = DataSet.TensorDataset(torch.IntTensor(np.array(trainX, dtype=int)),
                                 torch.IntTensor(np.array(trainY, dtype=int)))
# ✅ 修复1: num_workers=0 Windows必须，变量名 train_ds 不是 train_dataset
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

# 校验数据
valid_ds = DataSet.TensorDataset(torch.IntTensor(np.array(validX, dtype=int)),
                                 torch.IntTensor(np.array(validY, dtype=int)))
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
# %%
'''
实现一个简单的RNN，其构架主要包含3层：输入层，一层隐含层和输出层
'''


class SimpleRNN(nn.Module):
    def __init__(self, output_size, word_num, embedding_size, hidden_size, num_layers=1):
        # 定义
        super(SimpleRNN, self).__init__()

        # 一个embedding层
        self.embedding = nn.Embedding(word_num, embedding_size)

        # PyTorch的RNN层，batch_first标识可以让输入的张量的第一个维度表示batch指标
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True)

        # 输出的全连接层
        self.fc = nn.Linear(hidden_size, output_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x, hidden):
        # 运算过程
        # 先进行embedding层的计算
        x = self.embedding(x)
        # 从输入到隐含层的计算
        # x的尺寸为：batch_size，num_step，hidden_size
        output, hidden = self.rnn(x, hidden)
        # 从output中去除最后一个时间步的数值（output中包含了所有时间步的结果）
        output = output[:, -1, :]
        # output的尺寸为：batch_size，hidden_size
        # 最后一层全连接网络
        output = self.fc(output)
        # output的尺寸为：batch_size，output_size
        return output, hidden

    def initHidden(self, batch_size):
        # 对隐含单元初始化
        # 尺寸是layer_size，batch_size，hidden_size
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))


# %%
# 计算预测错误率的函数
def accuracy(pre, label):
    pre = torch.max(pre.data, 1)[1]
    rights = pre.eq(label.data).sum()
    acc = rights.data / len(label)
    return acc.float()


# %%
# 模型验证
def validate(model, val_loader, criterion, use_cuda):
    val_loss = 0
    val_acc = 0
    model.eval()
    for batch, data in enumerate(val_loader):
        init_hidden = model.initHidden(len(data[0]))
        if use_cuda:
            init_hidden = init_hidden.cuda()
        x, y = Variable(data[0]), Variable(data[1])
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        outputs, hidden = model(x, init_hidden)
        y = y.long()
        loss = criterion(outputs, y)
        val_loss += loss.data.cpu().numpy()
        val_acc += accuracy(outputs, y)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


# %%
# 打印训练结果
def print_log(epoch, train_time, train_loss, train_acc, val_loss, val_acc, epochs=10):
    print(
        f"Epoch [{epoch}/{epochs}], time: {train_time:.2f}s, loss: {train_loss:.4f}, acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")


# %%
# 定义主函数：模型训练
def train(model, optimizer, train_loader, val_loader, criterion, epochs=1, use_cuda=True):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        start = time.time()
        for batch, data in enumerate(train_loader):
            model.train()
            init_hidden = model.initHidden(len(data[0]))
            if use_cuda:
                init_hidden = init_hidden.cuda()
            optimizer.zero_grad()
            x, y = Variable(data[0]), Variable(data[1])
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            outputs, hidden = model(x, init_hidden)
            y = y.long()
            loss = criterion(outputs, y)
            train_loss += loss.data.cpu().numpy()
            train_acc += accuracy(outputs, y)
            loss.backward()
            optimizer.step()

        end = time.time()
        train_time = end - start
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, use_cuda)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print_log(epoch + 1, train_time, train_loss, train_acc, val_loss, val_acc, epochs=epochs)
    return train_losses, train_accs, val_losses, val_accs


# %%
# ✅ 修复2：所有训练代码都放在 main 保护下
if __name__ == '__main__':

    # 自动检测是否有GPU
    use_cuda = torch.cuda.is_available()
    print(f"使用GPU: {use_cuda}")

    # 获取文本数据集中包含的字符数量
    vocab_size = len(word2idx.keys()) + 1

    # 给定超参数
    lr = 1e-3
    epochs = 20

    # ✅ 修复3：正确传参
    rnn = SimpleRNN(output_size=vocab_size, word_num=vocab_size,
                    embedding_size=64, hidden_size=128)

    if use_cuda:
        rnn = rnn.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    print(rnn)

    # 开始训练
    print("\n===== 开始训练 =====")
    history = train(rnn, optimizer, train_loader, valid_loader, criterion,
                    epochs=epochs, use_cuda=use_cuda)

    # %%
    # 使用RNN写藏头诗
    print("\n===== 生成藏头诗 =====")
    poem_incomplete = '床****前****明****月****光****'
    poem_index = []
    poem_text = ''

    for i in range(len(poem_incomplete)):
        current_word = poem_incomplete[i]

        if current_word != '*':
            index = word2idx[current_word]
        else:
            x = poem_index + [0] * (maxlen - 1 - len(poem_index))
            init_hidden = rnn.initHidden(1)
            if use_cuda:
                init_hidden = init_hidden.cuda()
            x = torch.IntTensor(np.array([x], dtype=int))
            x = Variable(x)
            if use_cuda:
                x = x.cuda()
            pre, hidden = rnn(x, init_hidden)
            pre = pre.cpu()
            index = torch.argmax(pre)
            current_word = [k for k, v in word2idx.items() if v == index][0]

        poem_index.append(index)
        poem_text = poem_text + current_word

    # 输出藏头诗
    print()
    print(poem_text[0:5])
    print(poem_text[5:10])
    print(poem_text[10:15])
    print(poem_text[15:20])
    print(poem_text[20:25])
# %%