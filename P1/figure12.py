import math
import random
import numpy as np
import loadFittingDataP1
from matplotlib import pyplot as plt
from scipy.misc import derivative
from scipy.optimize import check_grad

i = 0
bs = 1
X, y = loadFittingDataP1.getData()
def func(w):
    return np.linalg.norm(X.dot(w) - y)
def dfunc(w):
    return - 1.0 / len(X) * np.sum(X*(y-X.dot(w))[:, np.newaxis], axis=0)
def funci(w):
    global i, bs
    return np.linalg.norm(X[i:i+bs,:].dot(w) - y[i:i+bs])
def dfunci(w):
    global i, bs
    bX = X[i:i+bs,:]
    bY = y[i:i+bs]
    return - 1.0 / len(bX) * np.sum(bX*(bY-bX.dot(w))[:, np.newaxis], axis=0)

def grad_desc(func, dfunc, stepSize=0.001):
    w = np.random.normal(size=10)
    norms = []
    values = []
#    while np.linalg.norm(dfunc(w)) > 0.0001:
    for epoch in range(0, 10):
        values += [func(w)]
        norms += [np.linalg.norm(dfunc(w))]
        w -= stepSize * dfunc(w)
#        print(func(w), np.linalg.norm(dfunc(w)))
    return (values, norms, w)

def s_grad_desc():
    global i, bs
    t = 0
    w = np.random.normal(size=10)
    norms = []
    values = []
    #while np.linalg.norm(dfunc(w)) > 0.0001:
    for epoch in range(0, int(1*bs)):
        i = 0
        stepSize = (100000 + t) ** -0.5
        values += [func(w)]
        norms += [np.linalg.norm(dfunc(w))]
        for _ in range(int(len(X)/bs)):
            w -= stepSize * bs / len(X) * dfunci(w)
            i += bs
            values += [func(w)]
        t += 1
#        print(t, values[-1], np.linalg.norm(dfunc(w)))
    print(w)
    return (values, norms, w)

plt.figure(1)

bs = 100.0
y_values1, norms1, _ = grad_desc(func, dfunc)

bs = 50.0
y_values2, norms2, _ = s_grad_desc()

bs = 20.0
y_values3, norms3, _ = s_grad_desc()

bs = 1.0
y_values4, norms4, _ = s_grad_desc()

if True:
#    a, = plt.plot(y_values1[:], label="Batch (100)")
    b, = plt.plot(y_values2[:100], label="Mini (50)")
    c, = plt.plot(y_values3[:100], label="Mini (20)")
    d, = plt.plot(y_values4[:100], label="SGD (1)")
    plt.legend(handles=[b, c, d])
    plt.title('Fitting Data: MSE')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.savefig('figure10a.png')
else:
    e, = plt.plot(norms1, label="Batch (100)")
    f, = plt.plot(norms2, label="Mini (50)")
    g, = plt.plot(norms3, label="Mini (20)")
    h, = plt.plot(norms4, label="SGD (1)")
    plt.legend(handles=[a, b, c, d])
    plt.title('Fitting Data: Gradient Norm')
    plt.xlabel('epoch')
    plt.ylabel('norm')
    plt.savefig('figure10b.png')

plt.show()
