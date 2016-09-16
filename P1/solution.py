import math
import random
import numpy as np
import loadParametersP1
from matplotlib import pyplot as plt

def grad_desc(func, d_func):
    x = np.zeros(2)

    plot_x = []
    plot_y = []
    epoch = 1
    while np.linalg.norm(d_func(x)) > 0.00001:
        x -= 0.01 * d_func(x)
        plot_x += [epoch]
        plot_y += [func(x)]
        epoch += 1
        print(x)

    plt.xlabel('epoch')
    plt.ylabel('f(x)')
    plt.plot(plot_x, plot_y)
    plt.show()

    return x

gaussMean,gaussCov,quadBowlA,quadBowlb = loadParametersP1.getData()

def gfunc(x):
    diff = (x - gaussMean)
    icov = np.linalg.inv(gaussCov)
    norm = - 1.0 / math.sqrt((2 * 3.1415)**len(gaussMean) * np.linalg.norm(gaussCov))
    return norm * np.exp(- 0.5 * diff.transpose().dot(icov.dot(diff)))
def d_gfunc(x):
    diff = (x - gaussMean)
    icov = np.linalg.inv(gaussCov)
    return -gfunc(x) * icov.dot(diff)

def qfunc(x):
    return 0.5 * x.transpose().dot(quadBowlA.dot(x)) - x.transpose().dot(quadBowlb)
def d_qfunc(x):
    return quadBowlA.dot(x) - quadBowlb
def ad_qfunc(x):
    result = [
        (qfunc(x+np.array([0.1,0])) - qfunc(x-np.array([0.1,0]))) / 0.2,
        (qfunc(x+np.array([0,0.1])) - qfunc(x-np.array([0,0.1]))) / 0.2
    ]
    return np.array(result)

point = np.array([-1, 2.3242])
print(d_qfunc(point))
print(ad_qfunc(point))
print(grad_desc(qfunc, d_qfunc))
print(grad_desc(qfunc, ad_qfunc))
