import math
import lassoData
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso

val_x, val_y = lassoData.lassoValData()
test_x, test_y = lassoData.lassoTestData()
train_x, train_y = lassoData.lassoTrainData()

def transform(x):
    result = []
    for x_i in x:
        vector = [x_i]
        for m_i in range(1, 12 + 1):
            vector += [math.sin(0.4*math.pi*x_i*m_i)]
        result += [vector]
    return np.array(result)

def ridge_regression(x, y, c):
    u = np.mean(x, 0)
    z = x - u
    reg = c*np.eye(x.shape[1])
    w = np.linalg.inv(z.transpose().dot(z) + reg).dot(z.transpose()).dot(y)
    return (w, u)


clf = Lasso(alpha=0.01)
clf.fit(transform(train_x), train_y)
fake_x = np.array([i/100.0 for i in range(-100, 100)])
fake_y = clf.predict(transform(fake_x))
plt.plot(fake_x, fake_y)

plt.scatter(train_x, train_y, color='r')
plt.scatter(test_x, test_y, color='g')
plt.scatter(val_x, val_y, color='b')
plt.savefig(__file__.split('/')[-1] + '.png')
