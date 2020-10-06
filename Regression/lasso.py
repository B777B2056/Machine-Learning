import numpy as np
from sklearn.linear_model import Lasso

def get_data():
    data = np.genfromtxt('D:\\ML\\Lesson\\Regression\\data\\longley.csv', delimiter=',')
    x = data[1:, 2:]
    y = data[1:, 1]  # 前两行数据用于测试
    return x, y

def sklearn_lasso(x, y, lamda):
    lin_reg_regularization_l1 = Lasso(alpha=lamda)  # 创建回归模型（带L2惩罚项， 岭回归）
    lin_reg_regularization_l1.fit(x[1:], y[1:])
    print('采用sklearn包, 预测值为+{}, 实际值为+{}'.format(lin_reg_regularization_l1.predict(x[0, np.newaxis]), y[0]))

x_data, y_data = get_data()
sklearn_lasso(x_data, y_data, lamda=0.408)
