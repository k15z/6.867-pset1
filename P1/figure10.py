import math
import random
import numpy as np
import loadFittingDataP1
from matplotlib import pyplot as plt
from scipy.misc import derivative

def grad_desc(func, dfunc, stepSize=0.001, init=np.zeros(10)):
    global index
    w = init
    values = []
    while np.linalg.norm(dfunc(w)) > 0.00001:
        w -= stepSize * dfunc(w)
        values += [func(w)]
        print(func(w))
    return values

X, y = loadFittingDataP1.getData()
def func(w, batch=False):
    return np.linalg.norm(X.dot(w) - y)
def dfunc(w):
    return derivative(func, w)

y_values = grad_desc(func, dfunc)
plt.plot(y_values, label="Batch")
plt.title('Fitting Data')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()
