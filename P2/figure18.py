import numpy as np
import loadFittingDataP2
from matplotlib import pyplot as plt

def poly_basis(x, m):
    result = []
    for x_i in x:
        vector = [1.0]
        for m_i in range(1, m + 1):
            vector += [x_i**m_i]
        result += [vector]
    return np.array(result)

def ridge_regression(x, y, c):
    u = np.mean(x, axis=0)
    u[0] = 0
    z = x - u
    reg = c*np.eye(x.shape[1])
    w = np.linalg.inv(z.transpose().dot(z) + reg).dot(z.transpose()).dot(y)
    return (w, u)

x, y = loadFittingDataP2.getData(False)
for m in [2, 10]:
    plt.figure()
    handles = []
    for i in [0.0000000001, 0.05, 1.0]:
        order = m
        w, u = ridge_regression(poly_basis(x, order), y, i)
        sx = np.linspace(0.0,1.0,100)
        sy = (poly_basis(sx, order) - u).dot(w)
        handle, = plt.plot(sx, sy, label="lambda " + str(i))
        handles += [handle]
    plt.title("M = " + str(m))
    plt.legend(handles=handles)
    plt.scatter(x, y, color='b')
    plt.show()
