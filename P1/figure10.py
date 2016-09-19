import math
import random
import numpy as np
import loadFittingDataP1
from matplotlib import pyplot as plt
from scipy.misc import derivative
from scipy.optimize import check_grad

i = 0
X, y = loadFittingDataP1.getData()
def func(w):
    return np.linalg.norm(X.dot(w) - y)
def dfunc(w):
    return - 1.0 / len(X) * np.sum(X*(y-X.dot(w))[:, np.newaxis], axis=0)
def funci(w):
    global i
    return np.linalg.norm(X[i,:].dot(w) - y[i])
def dfunci(w):
    global i
    return - X[i,:] * (y[i]-X[i,:].dot(w))

def grad_desc(func, dfunc, stepSize=0.001, init=np.random.normal(size=10)):
    w = init
    norms = []
    values = []
#    while np.linalg.norm(dfunc(w)) > 0.0001:
    for epoch in range(0, 20):
        values += [func(w)]
        norms += [np.linalg.norm(dfunc(w))]
        w -= stepSize * dfunc(w)
#        print(func(w), np.linalg.norm(dfunc(w)))
    return (values, norms, w)

def s_grad_desc(init=np.random.normal(size=10)):
    global i
    t = 0
    w = init
    norms = []
    values = []
    #while np.linalg.norm(dfunc(w)) > 0.0001:
    for epoch in range(0, 20):
        i = 0
        stepSize = (100000 + t) ** -0.5
        values += [func(w)]
        norms += [np.linalg.norm(dfunc(w))]
        for _ in range(len(X)):
            w -= stepSize / len(X) * dfunci(w)
            i += 1
        t += 1
#        print(t, values[-1], np.linalg.norm(dfunc(w)))
    print(w)
    return (values, norms, w)

plt.figure(1)

y_values1, norms1, _ = grad_desc(func, dfunc)
y_values2, norms2, _ = s_grad_desc()

plt.subplot(211)
a, = plt.plot(y_values1, label="Batch")
b, = plt.plot(y_values2, label="SGD")
plt.legend(handles=[a, b])
plt.title('Fitting Data: MSE')
plt.xlabel('epoch')
plt.ylabel('mse')

plt.subplot(212)
c, = plt.plot(norms1, label="Batch")
d, = plt.plot(norms2, label="SGD")
plt.legend(handles=[a, b])
plt.title('Fitting Data: Gradient Norm')
plt.xlabel('epoch')
plt.ylabel('norm')

plt.tight_layout()
plt.show()
