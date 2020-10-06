'''多元线性回归， 标准方程法求参数'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''导入数据'''
data = np.genfromtxt('D:\\ML\\Lesson\\Regression\\data\\Delivery.csv', delimiter=',')
x = data[:, :2]
Y = data[:, 2]
X = np.insert(np.array(x), 0, 1, 1)

'''标准方程法计算'''
def std_equ(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T  # 转换为列向量
    xTx = x_mat.T * x_mat
    # xTx不满秩则xTx不可逆，标准方程法不可用
    if np.linalg.matrix_rank(xTx) < len(x_arr[0]):
        print('xTx不可逆，标准方程法不可用')
        return
    return xTx.I * x_mat.T * y_mat

theta = np.array(std_equ(X, Y))
print(theta)
# 创建画布
fig = plt.figure(figsize=(12, 8),
                 facecolor='lightyellow'
                )

# 创建 3D 坐标系
ax = fig.gca(fc='whitesmoke',
             projection='3d'
            )

# 二元函数定义域平面
x_n = np.linspace(0, 100, 9)
y_n = np.linspace(0, 9, 9)
X_n, Y_n = np.meshgrid(x_n, y_n)
for i in range(0, len(x)):
    ax.scatter(x[i][0], x[i][1], Y[i])
ax.plot_surface(X=X_n,
                Y=Y_n,
                Z=theta[0][0]+theta[1][0]*X_n+theta[2][0]*Y_n,
                color='y',
                alpha=0.6
               )
plt.show()