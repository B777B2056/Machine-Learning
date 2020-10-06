'''三层BP神经网络， 输入层有d个输入， 隐含层有q个神经元， 输出层有l个输出'''
import numpy as np
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

class BP:
    def __init__(self, d, l, q, max_error=0.01, learn_rate=0.1, max_depth=6000):
        self.learn_rate = learn_rate
        self.d = d
        self.l = l
        self.q = q
        self.max_depth = max_depth  # 设置最大迭代深度，防止在陷入梯度下降缓慢时仍继续迭代
        self.max_error = max_error
        self.W = None
        self.V = None
        self.theta = None
        self.gama = None

    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    '''标准BP算法'''
    def train_standard(self, x_data, y_data):
        X = np.array(x_data)  # nxd
        n = X.shape[0]
        Y = np.array(y_data)  # nxl
        Y = Y.reshape((n, self.l))
        depth = 0
        # 参数随机初始化
        V = np.random.random((self.d, self.q))  # dxq
        W = np.random.random((self.q, self.l))  # qxl
        gama = np.random.random((1, self.q))  # 1xq
        theta = np.random.random((1, self.l))  # 1xl
        # 迭代
        while depth < self.max_depth:
            Error = []
            for i in range(n):
                # 误差前向传播
                Alpha = np.dot(X[i], V)  # 1xq
                B = self.__sigmoid(Alpha - gama)  # 1xq
                Beta = np.dot(B, W)  # 1xl
                Y_k = self.__sigmoid(Beta - theta)  # 1xl
                Error_k = 0
                for j in range(0, self.l):
                    Error_k += 0.5 * ((Y_k-Y[i])[0][j]) ** 2
                Error.append(Error_k)
                # 误差反向传播(单个样本误差过大则更新权重)
                g = Y_k * (1-Y_k) * (Y[i]-Y_k)  # 1xl
                e = B * (1-B) * np.dot(g, W.T)  # 1xq
                chang_V = self.learn_rate * np.dot(X[i].reshape((len(X[i]), 1)), e.reshape(1, self.q))  # dxq
                chang_W = self.learn_rate * np.dot(B.reshape((self.q, 1)), g.reshape(1, self.l))   # qxl
                chang_gama = -self.learn_rate * e  # 1xq
                change_theta = -self.learn_rate * g  # 1xl
                V += chang_V
                W += chang_W
                gama += chang_gama
                theta += change_theta
            depth += 1
            if sum(Error)/n <= self.max_error:
                break
        self.W = W
        self.V = V
        self.theta = theta
        self.gama = gama

    '''累积BP算法'''
    def train_accumulation(self, x_data, y_data):
        X = np.array(x_data)  # nxd
        n = X.shape[0]
        Y = np.array(y_data)  # nxl
        Y = Y.reshape((n, self.l))
        depth = 0
        # 参数随机初始化
        V = np.random.random((self.d, self.q))  # dxq
        W = np.random.random((self.q, self.l))  # qxl
        gama = np.random.random((1, self.q))  # 1xq
        theta = np.random.random((1, self.l))  # 1xl
        # 迭代
        while depth < self.max_depth:
            # 误差前向传播
            Alpha = np.dot(X, V)  # nxq
            B = self.__sigmoid(Alpha-gama)  # nxq
            Beta = np.dot(B, W)  # nxl
            Y_k = self.__sigmoid(Beta - theta)  # nxl
            Error = 0
            for i in range(n):
                for j in range(self.l):
                    Error += 0.5 * ((Y_k[i] - Y[i])[j]) ** 2
            Error /= n
            if Error < self.max_error:
                break
            # 误差反向传播(样本总体误差过大则更新权重)
            g_n = Y_k * (1 - Y_k) * (Y - Y_k)  # nxl
            e_n = B * (1 - B) * np.dot(g_n, W.T)  # nxq
            chang_V = self.learn_rate * np.dot(X.reshape((self.d, n)), e_n.reshape(n, self.q))  # dxq
            chang_W = self.learn_rate * np.dot(B.reshape((self.q, n)), g_n.reshape(n, self.l))  # qxl
            chang_gama = -self.learn_rate * e_n.sum(axis=0)  # 1xq, e_n.sum(axis=0)意为e_n维度降为一行，各元素为原矩阵对应列之和
            change_theta = -self.learn_rate * g_n.sum(axis=0)  # 1xl
            V += chang_V
            W += chang_W
            gama += chang_gama
            theta += change_theta
            depth += 1
        self.W = W
        self.V = V
        self.theta = theta
        self.gama = gama

    def predict(self, test_points):
        X = np.array(test_points)
        Alpha = np.dot(X, self.V)
        B = self.__sigmoid(Alpha-self.gama)
        beta = np.dot(B, self.W)
        Y_presict = self.__sigmoid(beta-self.theta)
        return Y_presict

# 解决异或问题
def xor_solve():
    xor = BP(d=2, q=6, l=1)
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    xor.train_standard(x_data=x_train, y_data=y_train,)
    print(xor.predict(x_train))

# Iris数据集
def iris():
    # 原始数据处理与数据集分割
    x_data = np.genfromtxt('Iris.csv', delimiter=',')
    y_data = np.genfromtxt('Iris.csv', delimiter=',', dtype=str)
    X = x_data[1:, 1:5]
    Y = []
    y = y_data[1:, 5]
    for i in range(len(y)):
        if y[i] == 'Iris-setosa':
            Y.append([1, 0, 0])
        elif y[i] == 'Iris-versicolor':
            Y.append([0, 1, 0])
        else:
            Y.append([0, 0, 1])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # 训练
    iris_train = BP(d=4, q=5, l=3, max_error=0.01)
    iris_train.train_standard(x_data=X_train, y_data=Y_train)
    # 预测
    output = []
    o = iris_train.predict(test_points=X_test)
    for i in range(len(o)):
        max_possible = max(o[i])
        for j in range(len(o[i])):
            if max_possible == o[i][j]:
                output.append(j+1)
                break
    print(output)
    # 比对
    real = []
    for i in range(len(Y_test)):
        for j in range(len(Y_test[i])):
            if Y_test[i][j] == 1:
                real.append(j+1)
    print(real)
    acc = 0.0
    for i in range(len(output)):
        if output[i] == real[i]:
            acc += 1
    print('测试集准确率为：{}%'.format(100 * acc / len(output)))

# EEG情感数据集（10折交叉验证）
def emotions_CV():
    start = time.clock()
    '''原始数据处理'''
    x_data = np.genfromtxt('emotions.csv', delimiter=',')
    y_data = np.genfromtxt('emotions.csv', delimiter=',', dtype=str)
    X = x_data[:, :x_data.shape[1]-1]
    X = StandardScaler().fit_transform(X)  # 特征标准化为单位比例（均值=0，方差=1）
    pca = PCA(n_components=20)
    X = pca.fit_transform(X)
    Y = []
    y = y_data[:, y_data.shape[1]-1]
    '''配置标签'''
    for i in range(len(y)):
        if y[i] == 'NEGATIVE':
            Y.append([1, 0, 0])
        elif y[i] == 'NEUTRAL':
            Y.append([0, 1, 0])
        else:
            Y.append([0, 0, 1])
    '''10折交叉验证数据处理, 随机等分为10份'''
    k = 1
    accuracy = []
    kf = KFold(n_splits=10)
    '''10折交叉验证'''
    for train_index, test_index in kf.split(X):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for i in range(len(train_index)):
            X_train.append(X[train_index[i]])
            Y_train.append(Y[train_index[i]])
        for i in range(len(test_index)):
            X_test.append(X[test_index[i]])
            Y_test.append(Y[test_index[i]])
        '''训练'''
        emotions_train = BP(d=X.shape[1], q=int(X.shape[1] + 1), l=3, max_error=0.05)
        emotions_train.train_standard(x_data=X_train, y_data=Y_train)
        '''预测'''
        output = []
        o = emotions_train.predict(test_points=X_test)
        for i in range(len(o)):
            max_possible = max(o[i])
            for j in range(len(o[i])):
                if max_possible == o[i][j]:
                    if j == 0:
                        output.append(-1)
                    elif j == 1:
                        output.append(0)
                    else:
                        output.append(1)
                    break
        '''比对（交叉验证）'''
        real = []
        for i in range(len(Y_test)):
            for j in range(len(Y_test[i])):
                if Y_test[i][j] == 1:
                    if j == 0:
                        real.append(-1)
                    elif j == 1:
                        real.append(0)
                    else:
                        real.append(1)
        '''准确度计算'''
        acc = 0.0
        for i in range(len(output)):
            if output[i] == real[i]:
                acc += 1
        acc /= len(output)
        accuracy.append(acc)
        print('测试集准确率（交叉验证, 第{}折为测试集）为：{}%'.format(k, 100 * acc))
        k += 1
    end = time.clock()
    print('共耗时{}秒'.format(end-start))
    print('经过10折交叉验证后，该模型测试集平均精度为：{}%'.format(100 * np.mean(accuracy)))

# emotions_CV()
iris()
# xor_solve()
