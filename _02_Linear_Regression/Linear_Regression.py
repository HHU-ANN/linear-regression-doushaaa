# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os




try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    a1=0.00173
    x,y=read_data()
    T=np.eye(x.transpose().dot(x).shape[1])
    w=np.linalg.inv(x.transpose().dot(x)+a1*T).dot(x.transpose()).dot(y)
    return(data.dot(w))


def lasso(data):
   x,y=read_data()
   w=np.zeros((x.shape[1],1))
   a = 0.00000000009299999999999999999
   print(a)
   for i in range(15):
    dw=x.transpose().dot(x.dot(w)-y)-a*np.sign(w)
    w=w-a*dw
   ypre=data.dot(w)
   print(ypre)
   sum=0
   average=np.mean(ypre)
   return average


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


