import numpy as np
import loadFittingDataP2
from numpy.linalg import lstsq
from matplotlib import pyplot as plt

poly_order = 5
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

def grad_desc(func, dfunc):
    w = np.random.normal(size=x.shape[1])
    gradient = dfunc(w)
    iteration = 0
    while np.linalg.norm(gradient) > 0.00001:
        lr = 0.5
        w -= lr * gradient
        gradient = dfunc(w)
        if iteration % 100000 == 0:
            print(iteration, np.linalg.norm(gradient), func(w))
        iteration += 1
    return w
w = grad_desc(func, dfunc)
print(w)

# gradient descent least squares
raw_x = [i/100.0 for i in range(101)]
test_x = poly_basis(raw_x, poly_order)
test_y = test_x.dot(w)
plt.plot(raw_x,test_y)

# closed for least squares
w = lstsq(x, y)[0]
raw_x = [i/100.0 for i in range(101)]
test_x = poly_basis(raw_x, poly_order)
test_y = test_x.dot(w)
plt.plot(raw_x,test_y,"r--")

# original data points
x, y = loadFittingDataP2.getData(False)
plt.scatter(x, y)

plt.show()