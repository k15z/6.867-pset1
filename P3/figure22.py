import tqdm
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

def tune(l, m):
    order = m
    w, u = ridge_regression(poly_basis(ax, order), ay, l)
    sx = np.linspace(-2.9,2.45,10000)
    sy = (poly_basis(sx, order) - u).dot(w)

    ay_predicted_wathetoeush = (poly_basis(ax, order) - u).dot(w)
    training_mse = np.linalg.norm(ay_predicted_wathetoeush - ay)

    vy_predicted_wathetoeush = (poly_basis(vx, order) - u).dot(w)
    validation_mse = np.linalg.norm(vy_predicted_wathetoeush - vy)

    by_predicted_wathetoeush = (poly_basis(bx, order) - u).dot(w)
    testing_mse = np.linalg.norm(by_predicted_wathetoeush - by)

    return (training_mse, validation_mse, testing_mse)

m = 10
x_axis = np.linspace(0,0.5,20)
training_mses = []
validation_mses = []
for l in tqdm.tqdm(x_axis):
    training_mse, validation_mse, testing_mse = tune(l, m)
    training_mses += [training_mse]
    validation_mses += [validation_mse]

a, = plt.plot(x_axis, training_mses, label="Training")
b, = plt.plot(x_axis, validation_mses, label="Validation")
plt.title("L2 Regularization")
plt.xlabel("lambda")
plt.ylabel("error")
plt.legend(handles=[a, b])
plt.show()
