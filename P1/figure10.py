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
    values = []
    while np.linalg.norm(dfunc(w)) > 0.00001:
        w -= stepSize * dfunc(w)
        values += [func(w)]
        print(func(w))
    return values

def s_grad_desc(init=np.random.normal(size=10)):
    global i
    t = 0
    w = init
    values = []
    while np.linalg.norm(dfunc(w)) > 0.00001:
        i = 0
        stepSize = (50 + t) ** -0.75
        for _ in range(len(X)):
            w -= stepSize / len(X) * dfunci(w)
            i += 1
        values += [func(w)]
        t += 1
        print(t, values[-1])
    return values

#y_values = grad_desc(func, dfunc)
y_values = s_grad_desc()
plt.plot(y_values, label="Batch")
plt.title('Fitting Data')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()
