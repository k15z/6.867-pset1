import math
import lassoData
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

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


plt.figure(figsize=(9, 5))

clf = Lasso(alpha=0.1)
clf.fit(transform(train_x), train_y)
fake_x = np.array([i/100.0 for i in range(-100, 100)])
fake_y = clf.predict(transform(fake_x))
aa, = plt.plot(fake_x, fake_y, label="LASSO (0.1)")

clf = Ridge(alpha=0.1)
clf.fit(transform(train_x), train_y)
fake_x = np.array([i/100.0 for i in range(-100, 100)])
fake_y = clf.predict(transform(fake_x))
bb, = plt.plot(fake_x, fake_y, label="Ridge (0.1)")

clf = Lasso(alpha=0.0)
clf.fit(transform(train_x), train_y)
fake_x = np.array([i/100.0 for i in range(-100, 100)])
fake_y = clf.predict(transform(fake_x))
cc, = plt.plot(fake_x, fake_y, label="Unregularized")

fake_x = np.array([i/100.0 for i in range(-100, 100)])
fake_y = transform(fake_x).dot(np.loadtxt('lasso_true_w.txt'))
dd, = plt.plot(fake_x, fake_y, 'c--', label="Actual")

a = plt.scatter(train_x, train_y, color='b', label="Training")
b = plt.scatter(test_x, test_y, color='g', label="Testing")
c = plt.scatter(val_x, val_y, color='r', label="Validation")
plt.legend(handles=[a, b, c, aa, bb, cc, dd], loc=2)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(__file__.split('/')[-1] + '.png')
