# coding:utf-8
import pandas as pd
from sklearn.decomposition import PCA
from src.WordEmbedding.w2v import w2v
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class DRV:
    def __init__(self, data, dim, path):
        self.path = path
        self.data = data
        self.dim = dim
        self.estimator = PCA(n_components=self.dim)
        self.ax = plt.subplot(111, projection='3d')
        self.w2v = w2v(path)
        self.w2v_word = self.w2v.word_vec(self.data)

    def reduce(self):
        result = self.estimator.fit_transform(self.w2v_word)
        return result

    def scatter_pic(self):
        data_embedding = self.reduce()
        data_embedding = np.array(data_embedding)
        xs = data_embedding[:, 0]
        ys = data_embedding[:, 1]
        zs = data_embedding[:, 2]
        self.ax.scatter(xs, ys, zs, c='y')
        for i in range(len(self.data)):
            self.ax.text(xs[i], ys[i], zs[i], self.data[i])

        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

        plt.show()

df = pd.read_csv('../data/result.txt')
path = "D:\project\model\word2vec/200w.bin"
data = ['开心', '天天', '天天开心', '真好', '快乐', '兴奋', '昨天', '明天']
d = DRV(list(df['word']), 3, path)
#d = DRV(data, 3, path)
d.scatter_pic()