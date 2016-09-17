import math
import random
import numpy as np
import loadParametersP1
from matplotlib import pyplot as plt

def grad_desc(func, d_func, stepSize, init=np.zeros(2)):
    x = init
    values = []
    for epoch in range(0,5000):
        print stepSize*d_func(x)
        x -= stepSize * d_func(x)
        values += [func(x)]
    print(x)
    return values

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
    (.0001, 'c-'),
    (.001, 'r-'),
    (.01, 'g-'),
    (.1, 'b-')
]

handles = []
for pair in pairs:
    stepSize, color = pair
    val = str(stepSize)
    y_values = grad_desc(qfunc, d_qfunc, stepSize, init=[-100,100])
    handles += [plt.plot(y_values[90:], color, label=val)[0]]
plt.legend(handles=handles)
plt.title('Gradient Descent: Quadratic')
plt.xlabel('epoch')
plt.ylabel('f(x)')
plt.show()