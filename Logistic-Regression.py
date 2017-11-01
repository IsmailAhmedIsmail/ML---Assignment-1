import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def hypothesis(theta, X):
    sum = np.dot(theta, X)
    return 1.0/(1.0 + math.exp(-1.0 * sum))


def cost(X, Y, theta, m):
    MSE = 0
    for i in range(m):
        if Y[i] == 0:
            e = math.log(1 - hypothesis(theta, X[i]))
        elif Y[i] == 1:
            e = math.log(hypothesis(theta, X[i]))
        MSE += e
    return (-1.0/m)*MSE


def gradient_descent(X, Y, theta, alpha, m):
    newTheta = []
    for j in range(len(theta)):
        sum = 0
        for i in range(m):
            sum += (hypothesis(theta, X[i]) - Y[i]) * X[i][j]
        value = theta[j] - (alpha/float(m)) * sum
        newTheta.append(value)
    return newTheta


def logisticRegression(X, Y, theta, alpha, m, num_iters):
    for i in range(num_iters):
        newTheta = gradient_descent(X, Y, theta, alpha, m)
        theta = newTheta
    return theta


iris = datasets.load_iris()
X = iris.data[:, :2]
X = np.insert(X,0,1,axis=1)
Y = iris.target

for i in range(len(Y)):
    if Y[i] == 2:
        Y[i] = 1

m = Y.size
iterations = 10000
alpha = 0.01
initial_theta = [0, 0, 0]

theta =  logisticRegression(X, Y, initial_theta, alpha, m, iterations)#To be completed by students
print ("Theta values: " + str(theta))
error = cost(X, Y, theta, m)
print ("Error: " + str(error))

plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

x = np.linspace(4, 8)
y = -(theta[0] + theta[1] * x)/theta[2]
plt.plot(x, y)

plt.show()