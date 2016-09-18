import math
import random
import numpy as np
import loadFittingDataP1
from matplotlib import pyplot as plt
from scipy.misc import derivative
from scipy.optimize import check_grad

X, y = loadFittingDataP1.getData()
def func(w):
    return np.linalg.norm(X.dot(w) - y)
def dfunc(w):
    return - 2.0 / len(X) * np.sum(X*(y-X.dot(w))[:, np.newaxis], axis=0)

def grad_desc(func, dfunc, stepSize=0.001, init=np.random.normal(size=10)):
    w = init
    values = []
    while np.linalg.norm(dfunc(w)) > 0.00001:
        w -= stepSize * dfunc(w)
        values += [func(w)]
        print(func(w))
    return values

y_values = grad_desc(func, dfunc)
plt.plot(y_values, label="Batch")
plt.title('Fitting Data')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()
