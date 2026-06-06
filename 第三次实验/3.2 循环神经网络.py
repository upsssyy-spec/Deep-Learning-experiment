
# 代码3-23
# 使用Embedding类构建训练矩阵
import torch
from torch.autograd import Variable
from torch import nn
"""
word_to_id = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 10)
hello_idx = torch.LongTensor([word_to_id['hello']])
hello_embed = embeds(hello_idx)
print(hello_embed)



# 代码3-24 
# 使用SimpleRNN类nn.RNN构建网络
x = Variable(torch.randn(6, 5, 100)) # 这是rnn的输入格式
rnn_seq = nn.RNN(100, 200)
print(rnn_seq.weight_hh_l0)  # 与h相乘的权重
print(rnn_seq.weight_ih_l0)  # 与x相乘的权重
out, h_t = rnn_seq(x)  # 使用默认的全 0 隐藏状态
h_0 = Variable(torch.randn(1, 5, 200))
out, h_t = rnn_seq(x, h_0)
print(out.shape, h_t.shape)



# 代码3-25 
# 使用SimpleRNN类nn.RNNCell构建网络
cell = nn.RNNCell(100, 20)
x = torch.randn(3, 100)
xs = [torch.randn(3, 100) for i in range(10)]
h = torch.zeros(3, 20)
for xt in xs:
    h = cell(xt, h)
print(h.shape)



# 代码 3-26 
# 使用LSTM构建网络
batch_size = 10  # 句子的数量
seq_len = 20  # 每个句子的长度
embedding_dim = 30  # 用长度为30的向量表示一个词语
word_vocab = 100  # 词典的数量
hidden_size = 18  # 隐藏层中lstm的个数
num_layer = 2  # 多少个隐藏层
in_put = torch.randint(low=0, high=100, size=(batch_size, seq_len))
# 把embedding之后的数据传入lstm
embedding = torch.nn.Embedding(word_vocab, embedding_dim) 
lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layer)
embed = embedding(in_put)  # [10, 20, 30]
embed = embed.permute(1, 0, 2)  # [20, 10, 30]
h_0 = torch.rand(num_layer, batch_size, hidden_size)
c_0 = torch.rand(num_layer, batch_size, hidden_size)
out_put, (h_1, c_1) = lstm(embed, (h_0, c_0))
print(out_put.size())
print(h_1.size())
print(c_1.size())
last_output = out_put[-1, :, :]
print(last_output.size())
last_hidden_state = h_1[-1, :, :]
print(last_hidden_state.size())



# 代码 3-27 
# 使用GRU构建网络
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2)
print(gru._parameters.keys())
print(gru.weight_ih_l0.shape)
print(gru.weight_hh_l0.shape)
"""


# 代码 3-28 
# 加载数据并预处理
import numpy as np
import pandas as pd

data_csv = pd.read_csv('./data.csv', usecols=[1])
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))




# 代码 3-29 
# 创建数据集
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i: (i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
# 创建好输入输出
data_X, data_Y = create_dataset(dataset)
# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[: train_size]
train_Y = data_Y[: train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

    
    
    
# 代码3-30 
# 改变数据形状
train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)



# 代码3-31 
# 构建LSTM网络
class lstm_reg(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x




# 代码3-32 
# 构建RNN网络
net = lstm_reg(2, 4)
criterion = nn.MSELoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)  # Adam优化算法



# 代码3-33 
# 训练网络
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e ) % 100 == 0:  # 每100次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e, loss))




# 代码3-34
# 性能评估
# 用训练好的模型去预测结果
import matplotlib.pyplot as plt
plt.rcParams['font.family']='SimHei'

net = net.eval()
data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data)  # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='预测值')
plt.plot(dataset, 'b', label='真实值')
plt.legend(loc='best')
plt.show()



















