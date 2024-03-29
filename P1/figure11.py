import math
import random
import numpy as np
import loadParametersP1
from matplotlib import pyplot as plt

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

def approx_deriv(func, x, delta):
    result = [
        (func(x+np.array([delta,0])) - func(x-np.array([delta,0]))) / (2*delta),
        (func(x+np.array([0,delta])) - func(x-np.array([0,delta]))) / (2*delta)
    ]
    return np.array(result)

point = [16,5]
print d_qfunc(point)
for delta in [100, 10, 1, .1]:
    print approx_deriv(qfunc, point, delta)