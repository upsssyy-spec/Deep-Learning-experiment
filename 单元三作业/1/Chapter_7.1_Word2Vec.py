#%%
# 代码7-1
import numpy as np
import json
# 数据路径及词典路径
path = 'dataset/imdb.npz'
dict_path = 'dataset/imdb_word_index.json'
#读入原始数据（这里的原始数据实际已经是编码成数字后的形式）
with np.load(path, allow_pickle=True) as f:
    x_train, labels_train = f['x_train'], f['y_train']
    x_test, labels_test = f['x_test'], f['y_test']

# 读取 词语-数字编码 字典
with open(dict_path) as f:
    word_index = json.load(f)
# 将其反转成 数字编码-词语
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 将原始的数字编码还原成英文单词
decoded_review = [ ' '.join([reverse_word_index.get(i , '?') for i in line]) for line in x_train]
#展示部分影评数据
print(decoded_review[0:5])

#%%
# 代码7-2：进行分词
train_data = []
for line in decoded_review:
    line_fenci = list(line.split(' '))
    train_data.append(line_fenci)

#%%
# 代码7-3：展示分词结果
print(train_data[30])
print(train_data[31])
print(train_data[32])

#%%
# 代码7-4：使用gensim中的Word2Vec
from gensim.models import Word2Vec
# 训练Word2Vec模型，size为词向量的维度，词频小于min_count的词将不被考虑
model = Word2Vec(train_data, vector_size=100, min_count=1)

#%%
# 代码7-5：查看词向量的长度
print(len(model.wv['awful']))
# 展示‘awful’这一单词的词向量
print(model.wv['awful'])

#%%
# 代码7-6：分别查看“good”与“bad”,“good”与“movie”两组词的相似性
print(model.wv.similarity('good', 'bad'))
print(model.wv.similarity('good', 'movie'))
# 查看与“awful”相关性大小排名前5的词语
for key in model.wv.similar_by_word('awful', topn=5):
    print(key)

#%%
# 代码7-7：绘制星空图
# 获得所有词语的列表
word_list = [reverse_word_index.get(i, '?') for i in range(1, 88585)]
# 通过模型获得所有词语的词向量构成的矩阵
X = model.wv[word_list]

# 原始的词向量维度过高，为可视化展示，需要降维，这里使用PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()  # 标准化
X = X_scaler.fit_transform(X)
# PCA
pca = PCA(n_components=2)  # 降为2维
pca.fit(X)
X_reduced = pca.transform(X)

# 导入绘图包
import matplotlib.pyplot as plt
# 首先对每个词语都绘制一个点
fig = plt.figure(figsize=(30, 15))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:,1], '.', markersize=1, alpha=0.1, color='white')
ax.set_xlim([-33, 30])
ax.set_ylim([-17, 40])

#选择几个特殊的词，不仅画出它们的位置，而且也把与其距离近（即相关性强）的词语画出来
words = ['bad', 'director', 'zombie']
all_words = []
for w in words:
    # 获取与指定词相关性最高的6个词
    lst = model.wv.similar_by_word(w, topn =6)
    wds = [i[0] for i in lst]
    wds.append(w)
    all_words.append(wds)

# 对每组词语分别指定颜色进行绘制
colors = ['red', 'yellow', 'cyan', 'green', 'orange']
for num, wds in enumerate(all_words):
    for w in wds:
        ind = word_index[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha=1, color=colors[num])
        # 将文本也标记在图上
        plt.text(xy[0], xy[1], w, fontsize=20, alpha=1, color=colors[num])

plt.show()