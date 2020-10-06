'''感知机学习模型'''
import numpy as np
import matplotlib.pyplot as plt

class SinglePerceptron:
    # 参数初始化
    def __init__(self, w_init=None, a_init=None, b_init=0, learn_rate=1):
        if a_init is None:
            a_init = [0, 0, 0]
        if w_init is None:
            w_init = [0, 0]
        self.w0 = np.array(w_init)
        self.a0 = np.array(a_init)
        self.b0 = b_init
        self.learn_rate = learn_rate
        self.W = None
        self.b = None

    # 原始形式学习算法
    def trainByOriginal(self, x_data, y_data):
        W = self.w0
        b = self.b0
        X = np.array(x_data)
        Y = np.array(y_data)
        haveWrong = True
        while haveWrong:
            wrong_cnt = 0
            print('W={}, b={}'.format(W, b))
            for i in range(0, len(X)):
                if Y[i] * (b + np.dot(W, X[i])) <= 0:
                    wrong_cnt += 1
                    W += self.learn_rate * Y[i] * X[i]
                    b += self.learn_rate * Y[i]
                    break
            if wrong_cnt == 0:
                haveWrong = False
        print('经由原始形式算法训练后，最终结果为：W={}, b={}'.format(W, b))
        self.W = W
        self.b = b

    # 对偶形式学习算法
    def trainByDual(self, x_data, y_data):
        A = self.a0
        b = self.b0
        X = np.array(x_data)
        Y = np.array(y_data)
        haveWrong = True
        # 计算gram矩阵
        gram = np.empty((len(X), len(X)), np.int)
        for i in range(0, len(X)):
            for j in range(0, len(X)):
                gram[i][j] = np.dot(X[i], X[j])
        print('===============================')
        print('Gram={}'.format(gram))
        print('===============================')
        while haveWrong:
            wrong_cnt = 0
            print('a={}, b={}'.format(A, b))
            for i in range(0, len(X)):
                sum_front = b
                for j in range(0, len(X)):
                    sum_front += A[j] * Y[j] * gram[i][j]
                if Y[i] * sum_front <= 0:
                    wrong_cnt += 1
                    A[i] += self.learn_rate
                    b += self.learn_rate * Y[i]
                    break
            if wrong_cnt == 0:
                haveWrong = False
        W = 0
        for i in range(0, len(X)):
            W += A[i] * Y[i] * X[i]
        print('对偶形式最终结果为：a={}, b={}'.format(A, b))
        self.W = W
        self.b = b

    # 预测与画图
    def predict(self, points):
        if self.W is None or self.b is None:
            print('模型未训练！')
            return
        for i in range(0, len(points)):
            plt.scatter(points[i][0], points[i][1], c='r')
            print('当前测试点坐标为{}, 预测结果为{}'.format(points[i], np.sign(self.b+np.dot(self.W, np.array(points[i])))))
        # 画分类用的线性超平面（退化为直线）
        r = np.linspace(-6, 6, 100)
        m = -(self.W[0] * r + self.b) / self.W[1]
        x_axis = [3, 4, 1]
        y_axis = [3, 3, 1]
        plt.scatter(x_axis, y_axis)
        plt.plot(r, m)
        plt.show()

# 由例2.1创建训练集
X = np.array([[3, 3], [4, 3], [1, 1]])
Y = np.array([1, 1, -1])
# 训练
sp = SinglePerceptron()
# sp.trainByOriginal(x_data=X, y_data=Y)
sp.trainByDual(x_data=X, y_data=Y)
# 预测
test_points = [[0, 0], [-5, 2], [3, 5]]
sp.predict(test_points)
