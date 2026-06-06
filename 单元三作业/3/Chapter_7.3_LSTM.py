# %%
# 导入相关模块
import os
import glob
import time
import subprocess
import pickle
import numpy as np
from pickle import dump, load
from music21 import converter, instrument, note, chord, stream

import torch
import torch.utils.data as DataSet
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader  # ✅ 修复：正确导入DataLoader

# 读取作曲任务所需序列数据
musicians = load(open('./dataset/LSTM/musicians', 'rb'))
namelist = load(open('./dataset/LSTM/namelist', 'rb'))
seqs = load(open('./dataset/LSTM/seqs', 'rb'))


# %%
# 定义序列编码函数
def seq_encode(seqs):
    seq2idx = {}
    seqs_digit = []

    i = 1
    for seq in seqs:
        for s in seq:
            if seq2idx.get(s) == None:
                seq2idx[s] = i
                i += 1

    for seq in seqs:
        seq_digit = []
        for s in seq:
            seq_digit.append(seq2idx[s])
        seqs_digit.append(seq_digit)
    return seq2idx, seqs_digit


seq2idx, seqs_digit = seq_encode(seqs)
print("原始序列")
print(seqs[123][1:100])
print("\n编码后的结果")
print(seqs_digit[123][1:100])


# %%
### 定义音乐家姓名编码函数
def musician_encode(namelist):
    name2idx = {}
    i = 0
    for name in namelist:
        if name2idx.get(name) == None:
            name2idx[name] = i
            i += 1

    namelist_digit = []
    for name in namelist:
        namelist_digit.append(name2idx[name])
    return name2idx, namelist_digit


name2idx, namelist_digit = musician_encode(namelist)
print("原始序列")
print(namelist[25:45])
print("\n编码后的结果")
print(namelist_digit[25:45])
# %%
# 将音乐家姓名编码转为one-hot形式
namelist_digit = F.one_hot(torch.tensor(namelist_digit))
print(f"音乐家one-hot形状: {namelist_digit.shape}")


# %%
### 定义生成训练输入输出序列函数
def generate_XY(seqs_digit, namelist, max_len):
    X = []
    Y = []
    i = -1
    for seq_digit in seqs_digit:
        i += 1
        if len(seq_digit) < 1:
            continue

        Y.append(seq_digit[-1])
        x = seq_digit[:-1] + [0] * (max_len - len(seq_digit))
        l = namelist_digit[i].tolist()
        X.append(x + l)
    idx = np.random.permutation(range(len(X)))
    X = [X[i] for i in idx]
    Y = [Y[i] for i in idx]
    return X, Y


max_len = 1000
X, Y = generate_XY(seqs_digit, namelist, max_len)
print("原始乐曲（部分）: ")
print(seqs[123][1:50])
print("变量X（音符序列）: ")
print(X[123][0:999][:20], "...")
print("变量X（作曲家）: ")
print(X[123][-9:])
print("变量Y: ")
print(Y[123])


# %%
### 定义一个LSTM模型类
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, output_size, word_num, embedding_size, hidden_size, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(word_num, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size

    def forward(self, x2, hidden, use_cuda=True):
        x = self.embedding(x2)
        if use_cuda:
            x = x.cuda()
        output, hidden = self.lstm(x, hidden)
        output = output[:, -1, ...]
        output = self.fc(output)
        return output

    def initHidden(self, x1, x1_size, batch_size, use_cuda=True):
        x = self.embedding(x1)
        if use_cuda:
            x = x.cuda()
            h1 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
            c1 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        else:
            h1 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            c1 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        _, out = self.lstm(x, (h1, c1))
        return out


# %%
### 定义预测准确率的函数
def accuracy(pre, label):
    pre = torch.max(pre.data, 1)[1]
    rights = pre.eq(label.data).sum()
    acc = rights.data / len(label)
    return acc.float()


# %%
### ✅ 修复致命Bug：x1和x2搞反了！
def split_x1_x2(x, use_cuda=True):
    x = x.tolist()
    # x1 = 前999位 = 音符序列
    # x2 = 后9位 = 音乐家one-hot
    x1 = [x[i][0:999] for i in range(len(x))]  # 音符序列
    x2 = [x[i][-9:] for i in range(len(x))]  # 音乐家one-hot
    x1 = torch.IntTensor(np.array(x1, dtype=int))
    x2 = torch.IntTensor(np.array(x2, dtype=int))
    if use_cuda:
        return Variable(x1).cuda(), Variable(x2).cuda()
    else:
        return Variable(x1), Variable(x2)


# %%
# 定义打印日志函数
def print_log(epoch, train_time, train_loss, train_acc, epochs=10):
    print(f"Epoch [{epoch}/{epochs}], time: {train_time:.2f}s, loss: {train_loss:.4f}, acc: {train_acc:.4f}")


# %%
### 定义模型训练函数
def train(model, optimizer, train_loader, epochs=1, use_cuda=True):
    train_losses = []
    train_accs = []

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        start = time.time()
        for batch, data in enumerate(train_loader):
            x, y = Variable(data[0]), Variable(data[1])
            if use_cuda:
                x, y = x.cuda(), y.cuda()

            # x1=音符, x2=音乐家
            x1, x2 = split_x1_x2(x, use_cuda)
            # 用音乐家one-hot初始化隐藏层
            init_hidden = model.initHidden(x2, 9, len(data[0]), use_cuda)
            optimizer.zero_grad()
            # 用音符序列做预测
            outputs = model(x1, init_hidden, use_cuda)

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
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print_log(epoch + 1, train_time, train_loss, train_acc, epochs=epochs)

    return train_losses, train_accs


# %%
### 定义生成音乐函数
def seq_to_mid(prediction):
    offset = 0
    output_notes = []
    for data in prediction:
        if ('.' in data) or data.isdigit():
            note_in_chord = data.split('.')
            notes = []
            for current_note in note_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')
    print("音乐已保存为 output.mid")


# %%
# ✅ Windows必须：所有训练代码放在main下
if __name__ == '__main__':

    # 自动检测GPU
    use_cuda = torch.cuda.is_available()
    print(f"使用GPU: {use_cuda}")

    # 设定batch size，Windows必须num_workers=0
    batch_size = 64
    ds = DataSet.TensorDataset(torch.IntTensor(np.array(X, dtype=int)),
                               torch.IntTensor(np.array(Y, dtype=int)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 获取数据集包含的音符数量
    seq_size = len(seq2idx.keys()) + 1
    lr = 1e-2
    epochs = 20  # 先跑20轮试试

    # 创建模型
    lstm = LSTMNetwork(input_size=max_len - 1, output_size=seq_size,
                       word_num=seq_size, embedding_size=256, hidden_size=128)

    if use_cuda:
        lstm = lstm.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
    print(lstm)

    # 开始训练
    print("\n===== 开始训练 =====")
    history = train(lstm, optimizer, loader, epochs=epochs, use_cuda=use_cuda)

    # %%
    ### 生成贝多芬的音乐
    print("\n===== 生成贝多芬风格音乐 =====")
    import random

    musicianname = 'beethoven'
    name_digit = name2idx[musicianname]
    name_digit = F.one_hot(torch.tensor(name_digit), num_classes=9)

    # 随机抽取贝多芬的一段乐曲
    input_index = []
    temp = []
    for i in range(len(seqs)):
        if namelist[i] == musicianname:
            temp = seqs_digit[i][0:20]
            vocab = list(seqs_digit[i])
            if random.random() > 0.5:
                input_index = seqs_digit[i][0:20]
                vocab = list(seqs_digit[i])
                break

    if len(input_index) == 0:
        input_index = temp
    input_index = list(input_index)

    # 生成500个音符
    # ===== 替换原来的生成部分代码 =====
    output_word = []
    length = 500

    # 🌡️ 温度参数：越小越保守，越大越随机（推荐0.5-0.8）
    temperature = 0.5

    for i in range(length):
        if i % 25 == 0:
            indexs = list(random.sample(vocab, 5))
            input_index.extend(indexs)
        else:
            x1 = input_index + [0] * (max_len - 1 - len(input_index))
            x1 = [int(i) if type(i) != int else i for i in x1]
            x1 = torch.IntTensor(np.array([x1], dtype=int))
            if use_cuda:
                x1 = Variable(x1).cuda()
            else:
                x1 = Variable(x1)

            x2 = torch.IntTensor(np.array([name_digit.tolist()], dtype=int))
            if use_cuda:
                x2 = Variable(x2).cuda()
            else:
                x2 = Variable(x2)

            init_hidden = lstm.initHidden(x2, 9, 1, use_cuda)
            pre = lstm(x1, init_hidden, use_cuda)
            pre = pre.cpu()

            # ✅ 关键改进：带温度的softmax采样，不是argmax
            pre = pre / temperature  # 温度控制
            probs = F.softmax(pre, dim=-1).squeeze()  # 转概率分布
            index = torch.multinomial(probs, 1).item()  # 按概率随机采样

            current_word = [k for k, v in seq2idx.items() if v == index][0]
            output_word.append(current_word)
            input_index.append(int(index))

    print(f"\n生成的前100个音符:")
    print(output_word[:100])

    # 转为MIDI文件
    seq_to_mid(output_word)
# %%