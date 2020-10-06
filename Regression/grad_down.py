'''一元线性回归， 梯度下降法'''
import numpy as np

'''数据载入与选取初始值'''
data = np.genfromtxt('D:\\ML\\Lesson\\Regression\\data\\data.csv', delimiter=',')
x = data[:, 0]  # 取第一列, 语法：list[行起始索引:行结束索引+1, 目标列数]
y = data[:, 1]  # 取第二列

'''损失函数'''
def loss(b, k):
    total_loss = 0
    for i in range(0, len(x)):
        total_loss += ((b + k * x[i] - y[i]) ** 2)
    total_loss /= (2.0 * float(len(x)))
    return total_loss

'''梯度下降法求使损失函数接近最小值的b、k'''
def grad_down(b, k, learn_rate, max_steps):
    m = float(len(x))
    for i in range(0, max_steps):
        grad_b = 0
        grad_k = 0
        for j in range(0, len(x)):
            grad_b += (b + k * x[j] - y[j])
            grad_k += ((b + k * x[j] - y[j]) * x[j])
        grad_b /= m
        grad_k /= m
        b -= (learn_rate * grad_b)
        k -= (learn_rate * grad_k)
        print('第'+str(i+1)+'次迭代hou，损失函数值为' + str(loss(b, k))+'\n')
    return b, k

'''测试用例'''
bs = 0  # 回归直线的初始截距
ks = 0  # 回归直线的初始斜率
lr = 0.0001  # 学习率(学习率的选取直接影响损失函数的输出，学习率太大会导致损失函数在迭代过程中上升)
b, k = grad_down(b=bs, k=ks, learn_rate=lr, max_steps=50)
print('b='+str(b)+', k='+str(k))

