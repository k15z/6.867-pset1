import math
import random
import numpy as np
import loadParametersP1
from matplotlib import pyplot as plt

def grad_desc(func, d_func, criteria=0.0001):
    x = np.array([-20.0,20.0])
    values = []
    epoch = 0
    while True:
        epoch += 1
        x -= 100 * d_func(x)
        values += [func(x)]
        if np.linalg.norm(d_func(x)) < criteria:
            print(str(x) + " " + str(criteria))
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

pairs = [
    (0.00000001, 'r-'),
    (0.0000001, 'r-'),
    (0.000001, 'r-'),
    (0.00001, 'r-'),
]

y_values = []
for pair in pairs:
    criteria, color = pair
    result = grad_desc(gfunc, d_gfunc, criteria=criteria)
    y_values += [np.linalg.norm(np.array([10, 10]) - result)]
plt.plot([x for x, y in pairs], y_values, 'b-')
plt.title('Gradient Descent: Gaussian')
plt.xlabel('threshold')
plt.ylabel('error')
plt.show()
