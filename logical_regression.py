import math
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import  *

def data_read(filename,k):
    """读入数据，当读训练数据时k取0，返回两个矩阵，测试数据返回一个矩阵"""
    ft = open(filename)
    lx = []
    ly1 = []
    ly2 = []
    ly3 = []
    read = ft.readlines()
    for line in read:
        te = list(line.rstrip('\n').split(','))
        if k == 0:
            if te[-1] == '1':
                ly1.append(1)
                ly2.append(0)
                ly3.append(0)
            if te[-1] == '2':
                ly1.append(0)
                ly2.append(1)
                ly3.append(0)
            if te[-1] == '3':
                ly1.append(0)
                ly2.append(0)
                ly3.append(1)
            te.pop()
        te_int = [float(x) for x in te]
        te_int.append(1.0)
        lx.append(te_int)
    mat_x = np.mat(lx)
    mat_y1 = np.mat(ly1).T
    mat_y2 = np.mat(ly2).T
    mat_y3 = np.mat(ly3).T
    if k == 0:
        return mat_x,mat_y1,mat_y2,mat_y3
    else:
        return mat_x

def sigmoid(a):
    m, n = a.shape
    for i in range(m):
            a[i, 0] = 1/(1+math.e**(-a[i, 0]))
    return a

def logical(mat_x,mat_y):
    count = 0#循环计数
    alpha = 0.001#步长
    loop_max = 10000#循环上限
    a,b = mat_x.shape
    w1 = np.mat(np.ones(b))#系数矩阵
    w1 = w1.T
    while count < loop_max:
        count += 1
        w2 =alpha*mat_x.T*(sigmoid(mat_x*w1)-mat_y)#梯度下降
        w1 = w1 - w2
    return w1
def output(w):
    m = len(w) -1
    for i in range(m):
        print(w[i, 0], end=' ')
    print(w[m, 0])

train_file = r"D:\train.txt"
test_file = r"D:\test.txt"
mat_x,mat_y1,mat_y2,mat_y3 = data_read(train_file,0)#训练数据
mat_test = data_read(test_file,1)#测试数据

ar_y = []#最后的预测结果
for i in range(len(mat_test)):
    ar_y.append('a')
w1 = logical(mat_x,mat_y1)
w2 = logical(mat_x,mat_y2)
w3 = logical(mat_x,mat_y3)
t1  = mat_test * w1
t2  = mat_test * w2
t3  = mat_test * w3
for i in range(len(t1)):
    if t1[i] > t2[i] and t1[i] > t3[i]:
        ar_y[i] = '1'
    elif t2[i] > t1[i] and t2[i] > t3[i]:
        ar_y[i] = '2'
    else:
        ar_y[i] = '3'
output(w1);output(w2);output(w3)
for i in range(len(ar_y)):
    print(ar_y[i])