import numpy as np
import regressData
from matplotlib import pyplot as plt

def poly_basis(x, m):
    result = []
    for x_i in x:
        vector = [1.0]
        for m_i in range(1, m + 1):
            vector += [x_i**m_i]
        result += [vector]
    return np.array(result)

order = 5
ax, ay = regressData.regressAData()
bx, by = regressData.regressBData()
vx, vy = regressData.validateData()

def ridge_regression(x, y, c):
    u = np.mean(x, 0)
    z = x - u
    reg = c*np.eye(x.shape[1])
    w = np.linalg.inv(z.transpose().dot(z) + reg).dot(z.transpose()).dot(y)
    return (w, u)

w, u = ridge_regression(poly_basis(ax, order), ay, 1.01)
print(w)

sx = np.array([i/100.0 for i in range(-290, 290)])
sy = (poly_basis(sx, order) - u).dot(w)
plt.plot(sx, sy)

plt.scatter(ax, ay, color='b')
plt.scatter(bx, by, color='r')
plt.scatter(vx, vy, color='g')
plt.show()