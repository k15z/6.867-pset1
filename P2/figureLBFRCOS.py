import math
import numpy as np
import loadFittingDataP2
from numpy.linalg import lstsq
from matplotlib import pyplot as plt

cos_order = 8
x, y = loadFittingDataP2.getData(False)

def cos_basis(x, m):
    result = []
    for x_i in x:
        vector = []
        for m_i in range(1, m + 1):
            vector += [math.cos(m_i*math.pi*x_i)]
        result += [vector]
    return np.array(result)
x = cos_basis(x, cos_order)

def func(w):
    return np.linalg.norm(x.dot(w) - y)

def dfunc(w):
    return - 1.0 / len(x) * np.sum(x*(y-x.dot(w))[:, np.newaxis], axis=0)

def grad_desc(func, dfunc):
    w = np.random.normal(size=x.shape[1])
    gradient = dfunc(w)
    iteration = 0
    while np.linalg.norm(gradient) > 0.00001:
        lr = 0.1
        w -= lr * gradient
        gradient = dfunc(w)
        if iteration % 100000 == 0:
            print(iteration, np.linalg.norm(gradient), func(w))
        iteration += 1
    return w
w = grad_desc(func, dfunc)
print("approx", w)

# gradient descent least squares
raw_x = [i/100.0 for i in range(101)]
test_x = cos_basis(raw_x, cos_order)
test_y = test_x.dot(w)
plt.plot(raw_x,test_y)

# closed for least squares
w = lstsq(x, y)[0]
print("actual", w)
raw_x = [i/100.0 for i in range(101)]
test_x = cos_basis(raw_x, cos_order)
test_y = test_x.dot(w)
plt.plot(raw_x,test_y,"r--")

# original data points
x, y = loadFittingDataP2.getData(False)
plt.scatter(x, y)

plt.show()
