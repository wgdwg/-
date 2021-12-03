#最小二乘法实现线性回归

import numpy as np #导入计算包
import matplotlib.pyplot as plt #导入绘图包
#1.加载数据源,并画出散点图
points = np.genfromtxt('data.csv', delimiter=',')

#提取points的两列数据，分别作为x,y
x = points[:,0] #第一列数据
y = points[:,1] #第2列数据

#用plt画出散点图
plt.scatter(x,y)
plt.show()

#2.定义损失函数,损失函数是系数的函数
def cost_function(w, b, points):
    total_const = 0 #初始化损失函数
    m = len(points) #数据个数
    #逐点计算平方误差，然后求平均数
    for i in range(m):
        x = points[i,0] #第i行，第1列
        y = points[i,1] #第i行，第2列
        total_const += (y - w * x - b) ** 2
    return total_const / m

#3.定义算法拟合函数

#平均值函数
def average(data):
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num

#定义核心拟合函数
def fit(points):
    m = len(points)
    x_bar = average(points[:,0]) #x的均值
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0

    for i in range(m):
        x = points[i, 0]  # 第i行，第1列
        y = points[i, 1]  # 第i行，第2列
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    #根据公式计算w
    w = sum_yx / (sum_x2 - m * (x_bar ** 2))

    for i in range(m):
        x = points[i, 0]  # 第i行，第1列
        y = points[i, 1]  # 第i行，第2列
        sum_delta += (y - w * x)
    #根据公式求b
    b = sum_delta / m

    return w, b

#4.测试
w, b = fit(points) #得到参数w,b
cost = cost_function(w,b,points) #得到损失函数
print("参数w = ", w)
print("参数b = ", b)
print("损失函数 = ", cost)

#5.画图拟合曲线
plt.scatter(x,y) #原始的散点图
pred_y = (w * x) + b #预测的y
plt.plot(x,pred_y,c='r') #红色的拟合直线
plt.show() #显示绘图