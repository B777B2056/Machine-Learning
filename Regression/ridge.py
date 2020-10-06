'''岭回归'''
import numpy as np
from sklearn.linear_model import Ridge

def get_data():
    data = np.genfromtxt('D:\\ML\\Lesson\\Regression\\data\\longley.csv', delimiter=',')
    x = data[1:, 2:]
    y = data[1:, 1]  # 前两行数据用于测试
    return x, y

def sklearn_rigde(x, y, lamda):
    lin_reg_regularization_l2 = Ridge(alpha=lamda)  # 创建回归模型（带L2惩罚项， 岭回归）
    lin_reg_regularization_l2.fit(x[1:], y[1:])
    print('采用sklearn包, 预测值为+{}, 实际值为+{}'.format(lin_reg_regularization_l2.predict(x[0, np.newaxis]), y[0]))

def std_equ_ridge(x, y, lamda):
    x_train = x[1:]
    y_arr = y[1:]
    x_arr = np.insert(np.array(x_train), 0, 1, 1)
    X = np.mat(x_arr)
    Y = np.mat(y_arr).T
    E = np.mat(np.identity(n=len(x_arr[0]), dtype=int))
    theta = (X.T*X + lamda*E).I * X.T * Y
    print('采用标准方程法, 预测值为+{}, 实际值为+{}'.format(np.array(np.mat(x_arr[1:])*theta)[0], y[0]))

x_data, y_data = get_data()
sklearn_rigde(x_data, y_data, lamda=0.408)
std_equ_ridge(x_data, y_data, lamda=0.408)
