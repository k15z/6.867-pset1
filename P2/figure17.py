import numpy as np
import loadFittingDataP2
from numpy.linalg import lstsq
from matplotlib import pyplot as plt

i = 0
bs = 1
poly_order = 3
x, y = loadFittingDataP2.getData(False)

def poly_basis(x, m):
    result = []
    for x_i in x:
        vector = [1.0]
        for m_i in range(1, m + 1):
            vector += [x_i**m_i]
        result += [vector]
    return np.array(result)
x = poly_basis(x, poly_order)

def func(w):
    return np.linalg.norm(x.dot(w) - y)
def dfunc(w):
    return - 1.0 / len(x) * np.sum(x*(y-x.dot(w))[:, np.newaxis], axis=0)
def funci(w):
    global i, bs
    return np.linalg.norm(x[i:i+bs,:].dot(w) - y[i:i+bs])
def dfunci(w):
    global i, bs
    bX = x[i:i+bs,:]
    bY = y[i:i+bs]
    return - 1.0 / len(bX) * np.sum(bX*(bY-bX.dot(w))[:, np.newaxis], axis=0)

def s_grad_desc():
    global i, bs
    t = 0
    w = np.random.normal(size=x.shape[1])
    if x.shape[1] == 11:
        w = np.array([  2.27480000e+00,  -7.92473167e+01,   2.44933527e+03,
            -3.23571022e+04,   2.18305603e+05,  -8.48291490e+05,
             2.00819859e+06,  -2.94128819e+06,   2.60119378e+06,
            -1.27236938e+06,   2.64236662e+05])
    norms = []
    values = []
    while np.linalg.norm(dfunc(w)) > 0.001:
        i = 0
        stepSize = (1 + t) ** -0.5
        values += [func(w)]
        norms += [np.linalg.norm(dfunc(w))]
        for _ in range(int(len(x)/bs)):
            w -= stepSize * bs / len(x) * dfunci(w)
            i += bs
        t += 1
        if t % 10000 == 0:
            print(t, norms[-1], values[-1])
    print(w)
    return (values, norms, w)

# gradient descent least squares
values, norms, w = s_grad_desc()
plt.plot(values)
plt.show()
