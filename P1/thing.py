import math
import random
import numpy as np
import loadParametersP1
from matplotlib import pyplot as plt

def gradient_descent(func, d_func, init=np.zeros(2)):
    x = init
    values = []
    for epoch in range(0,5000):
        x -= 100 * d_func(x)
        values += [func(x)]
    return values

gaussianMean, gaussianCovariance, _, _ = loadParametersP1.getData()
def gfunc(x):
    diff = (x - gaussianMean)
    icov = np.linalg.inv(gaussianCovariance)
    norm = - 1.0 / math.sqrt((2 * 3.1415)**len(gaussianMean) * np.linalg.norm(gaussianCovariance))
    return norm * np.exp(- 0.5 * diff.transpose().dot(icov.dot(diff)))
def d_gfunc(x):
    diff = (x - gaussianMean)
    icov = np.linalg.inv(gaussianCovariance)
    return -gfunc(x) * icov.dot(diff)

values = gradient_descent(gfunc, d_gfunc)
plt.plot(values, 'b-')
plt.xlabel('epoch')
plt.ylabel('f(x)')
plt.show()
