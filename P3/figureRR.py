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

ax, ay = regressData.regressAData()
bx, by = regressData.regressBData()
vx, vy = regressData.validateData()

def ridge_regression(x, y, c):
    u = np.mean(x, axis=0)
    u[0] = 0
    z = x - u
    reg = c*np.eye(x.shape[1])
    w = np.linalg.inv(z.transpose().dot(z) + reg).dot(z.transpose()).dot(y)
    return (w, u)

lambdas = [.00001, .0001, .001, .01, .1, 1, 10]
for i in lambdas:
    for m in range(1, 13):
        order = m
        w, u = ridge_regression(poly_basis(ax, order), ay, i)
        sx = np.linspace(-2.9,2.45,10000)
        sy = (poly_basis(sx, order) - u).dot(w)
        #plt.plot(sx, sy)
        print ("Lambda",i,"Order",m)
        ay_predicted_wathetoeush = (poly_basis(ax, order) - u).dot(w)
        print ("traing error" , np.linalg.norm(ay_predicted_wathetoeush - ay))
        vy_predicted_wathetoeush = (poly_basis(vx, order) - u).dot(w)
        print ("val error" , np.linalg.norm(vy_predicted_wathetoeush - vy))
        by_predicted_wathetoeush = (poly_basis(bx, order) - u).dot(w)
        print ("potato error" , np.linalg.norm(by_predicted_wathetoeush - by))
        print ("")

#plt.scatter(ax, ay, color='b')
#plt.scatter(bx, by, color='r')
#plt.scatter(vx, vy, color='g')

#plt.show()


