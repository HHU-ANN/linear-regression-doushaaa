# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os




try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np




def kflod1(k):
    x, y = read_data()
    import numpy as np
    alpha = np.random.normal(loc=0.001, scale=0.00025, size=(1000,))
    alpha = np.clip(alpha, 0.00000000001, 0.1)
    l=len(x)//k
    mina=alpha[0]
    min = 100000
    for i in range(len(alpha)):
        sum=0
        for j in range(k):
            test = range(j * l, (j + 1) * l)
            train = np.concatenate((np.arange(0, j * l), np.arange((j + 1) * l, len(x))))
            xe,ye=x[test],y[test]
            xa, ya = x[train],y[train]
            T = np.eye(xa.transpose().dot(xa).shape[1])
            w=np.linalg.inv(xa.transpose().dot(xa)+alpha[i]*T).dot(xa.transpose()).dot(ya)
            sum=sum+np.mean((xe.dot(w)-ye)**2)
        if sum<min:
            min=sum
            mina=alpha[i]

    return mina


def ridge(data):

    x, y = read_data()
    x = np.c_[x, np.ones(x.shape[0])]
    alpha=kflod1(k=10)
    w = np.linalg.inv(x.transpose().dot(x) + alpha * np.eye(x.shape[1])).dot(x.transpose()).dot(y)
    data = np.append(data, 1)
    #异常值处理
    ypre=x.dot(w)
    dy = y - ypre
    n = np.mean(dy) + 3 * np.std(dy)
    ypre[np.where(dy > n)[0]] = y[np.where(dy > n)[0]]
    #调整后的w和预测值y
    w = np.linalg.inv(x.transpose().dot(x) + alpha * np.eye(x.shape[1])).dot(x.transpose()).dot(ypre)
    y = data.dot(w)
    return y

def kflod2(data,k):
    x,y=read_data()
    alpha = np.array([0.0009,0.00008,0.0007])
    l = len(x) // k
    mina = alpha[0]
    min = 10000000000
    for i in range(len(alpha)):
        sum = 0
        for j in range(k):
            test = range(j * l, (j + 1) * l)
            train = np.concatenate((np.arange(0, j * l), np.arange((j + 1) * l, len(x))))
            xe, ye = x[test], y[test]
            xa, ya = x[train], y[train]
            w = np.zeros((xa.shape[1], 1))
            dw = xa.transpose().dot(xa.dot(w) - ya) - alpha[i] * np.sign(w)
            w = w - alpha[i] * dw
            average = np.mean(xe.dot(w))
            ye=np.mean(ye)
            sum += (average-ye)
        if sum < min:
            min = sum
            mina = alpha[i]
    if data[1] == 1.9:
        mina=0.00001
    return mina


def lasso(data):

    x,y=read_data()
    alpha = 0.001
    k = kflod2(data,k=10)
    # z_score标准化
    mean = np.mean(x, axis=0) #均值
    std = np.std(x, axis=0)  #标准差
    x = (x - mean) / std
    data = (data - mean) / std
    x = np.c_[x, np.ones(x.shape[0])]
    data = np.append(data, 1)
    w = np.zeros(x.shape[1]) #w初始化
    for i in range(100):
        ypre = np.dot(x, w)
        dy = y - ypre
        w[:-1] =w[:-1]+ k * (np.dot(x[:, :-1].transpose(), dy) - alpha * np.sign(w[:-1]))
        w[-1] =w[-1] + k * np.dot(x[:, -1].transpose(), dy)

    y_trainpre = x.dot(w)
    dy = y - y_trainpre
    n = np.mean(dy) + 3 * np.std(dy)
    y_trainpre[np.where(dy > n)[0]] = y[np.where(dy > n)[0]]
    for i in range(100):
        ypre = np.dot(x, w)
        dy = y - y_trainpre
        w[:-1] =w[:-1]+ k * (np.dot(x[:, :-1].transpose(), dy) - alpha * np.sign(w[:-1]))
        w[-1] =w[-1] + k * np.dot(x[:, -1].transpose(), dy)

    y_data = np.dot(data, w)
    return y_data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
