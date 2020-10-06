'''多元线性回归， 梯度下降法求参数'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''导入数据'''
data = np.genfromtxt('D:\\ML\\Lesson\\Regression\\data\\Delivery.csv', delimiter=',')
x_1 = data[:, 0]
x_2 = data[:, 1]
y = data[:, 2]
m = len(x_1)

'''损失函数'''
def loss(theta_0, theat_1, theta_2):
    total_loss = 0
    for i in range(0, m):
        total_loss += ((theta_0 + theat_1 * x_1[i] + theta_2 * x_2[i] - y[i]) ** 2)
    total_loss /= (2.0 * float(m))
    return total_loss

'''梯度下降法求使损失函数值尽可能小的参数theta_j(j=0, 1, 2)'''
def grad_down(theta_0, theat_1, theta_2, learn_rate, max_steps):
    for i in range(0, max_steps):
        grad_0 = 0
        grad_1 = 0
        grad_2 = 0
        for j in range(0, m):
            grad_0 += (theta_0 + theat_1 * x_1[j] + theta_2 * x_2[j] -y[j])
            grad_1 += (x_1[j] * (theta_0 + theat_1 * x_1[j] + theta_2 * x_2[j] -y[j]))
            grad_2 += (x_2[j] * (theta_0 + theat_1 * x_1[j] + theta_2 * x_2[j] -y[j]))
        grad_0 /= float(m)
        grad_1 /= float(m)
        grad_2 /= float(m)
        theta_0 -= learn_rate * grad_0
        theat_1 -= learn_rate * grad_1
        theta_2 -= learn_rate * grad_2
        print('第' + str(i + 1) + '次迭代后，损失函数值为' + str(loss(theta_0, theat_1, theta_2)) + '\n')
    return theta_0, theat_1, theta_2

'''测试用例'''
t_0, t_1, t_2 = grad_down(theta_0=0, theat_1=0, theta_2=0, learn_rate=0.0001, max_steps=1000)
print('theta_0='+str(t_0)+', theta_1='+str(t_1)+', theta_2='+str(t_2))
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
X, Y = np.meshgrid(x_n, y_n)
for i in range(0, len(x_1)):
    ax.scatter(x_1[i], x_2[i], y[i])
ax.plot_surface(X,
                Y,
                Z=t_0+t_1*X+t_2*Y,
                color='y',
                alpha=0.6
               )
plt.show()