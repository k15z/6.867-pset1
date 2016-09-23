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


width = 0.9

plt.figure(figsize=(15, 5))
coefs = np.loadtxt('lasso_true_w.txt')
a = plt.bar(np.array(range(len(coefs)))+width*0/4, coefs, width/4, color='g', label="Actual")

clf = Lasso(alpha=0.1)
clf.fit(transform(train_x), train_y)
b = plt.bar(np.array(range(len(clf.coef_)))+width*1/4, clf.coef_, width/4, color='c')

clf = Ridge(alpha=0.1)
clf.fit(transform(train_x), train_y)
c = plt.bar(np.array(range(len(clf.coef_[0])))+width*2/4, clf.coef_[0], width/4, color='b')

clf = Lasso(alpha=0.0)
clf.fit(transform(train_x), train_y)
d = plt.bar(np.array(range(len(clf.coef_)))+width*3/4, clf.coef_, width/4, color='r')

plt.ylabel('value')
plt.xlabel('coefficient')
plt.xticks(range(len(coefs)))
plt.legend([a[0], b[0], c[0], d[0]], ["Actual","Lasso", "Ridge", "Unregularized"])