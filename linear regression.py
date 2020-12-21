import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 首先我们需要采样自真实模型的多组数据，对于已知真实模型的玩具样例(Toy Example)， 我们直接从指定的𝑤 = 1.477, 𝑏 = 0.089的真实模型中直接采样
# # 生成数据
def generate_data(w,b):
    data = []
    # 100 data
    for i in range(100):
        x = np.random.uniform(-10, 10)
        eps = np.random.normal(0, 0.01) # guassian noise 均值0 方差0.01
        y = w*x + b + eps           # pretend to be the data with some noises
        data.append([x,y])
    data = np.array(data)
    return data

def plotData(data):
    '''
    :param data: (X,Y) = (100,2), ndarray
    :return: None
    '''
    X = data[0,:]
    Y = data[1,:]
    plt.scatter(X,Y,edgecolors='red')
    plt.show()

# MSE loss
def computeCost(y_hat,y):
    return

if __name__ == '__main__':
    data = generate_data(1.477,0.089)
    plotData(data)
