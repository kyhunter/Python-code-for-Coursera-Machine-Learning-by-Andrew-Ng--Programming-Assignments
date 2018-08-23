# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the second part
# %  of the exercise which covers regularization with logistic regression.
#
# Instructions copied from matlab code
# I am new python learner
# This code is for personal practice
# Discussions and suggestions are welcome

import numpy as np
import matplotlib.pylab as plt

data = np.loadtxt('ex2data2.txt', dtype=np.float, delimiter=',')
X = data[:, 0:-1]
y = data[:, 2]

def plotscatter(X_p, Y_p):
    pos = np.where(np.isin(Y_p, 1))
    pos_x = X_p[pos, 0]
    pos_y = X_p[pos, 1]
    neg = np.where(np.isin(Y_p, 0))
    neg_x = X_p[neg, 0]
    neg_y = X_p[neg, 1]
    axes = plt.subplot(111)
    pos_type = axes.scatter(pos_x, pos_y, c='black', s=40, marker='+')
    neg_type = axes.scatter(neg_x, neg_y, c='yellow', s=40, marker='o')
    plt.title('Chip test result')
    plt.xlabel('Chip test 1')
    plt.ylabel('Chip test 2')
    axes.legend((pos_type, neg_type), ('good', 'bad'), loc=2,)
    return


plotscatter(X, y)
plt.show()

# % MAPFEATURE Feature mapping function to polynomial features
# %
# %   MAPFEATURE(X1, X2) maps the two input features
# %   to quadratic features used in the regularization exercise.
# %
# %   Returns a new feature array with more features, comprising of
# %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
# %
# %   Inputs X1, X2 must be the same size
# %


def mapFeature(x1, x2):
    degree = 6
    out = np.array([np.ones(np.size(x1))])
    out = out.T
    for i in range(degree):
        for j in range(0, i+2):
            index = np.power(x1, (i + 1 - j)) * np.power(x2, j)
            if np.shape(index) is ():
                out = np.append(out, np.array([[index]]), axis=1)
            else:
                out = np.append(out, index, axis=1)
    return out


x_f1 = np.asarray([X[:, 0]])  # 1*118
x_f2 = np.asarray([X[:, 1]])

X = mapFeature(x_f1.T, x_f2.T)  # transpose the array to 118*1

initial_theta = np.zeros((np.shape(X)[1], 1))  # 28*1
lambda_re = 1


def sigmoid(z_s):
    g_s = np.zeros((np.size(z_s), 1))
    g_s = 1 / (1 + np.exp(-z_s))
    return g_s


def costFunctionReg(theta_c, X_c, y_c, lambda_c):
    m_temp = len(y_c)
    J_c = 0
    grad_c = np.zeros((len(theta_c), 1))
    J_c = ((1/m_temp) * np.sum(-np.transpose([y_c]) * np.log(sigmoid(np.dot(X_c, theta_c)))
                               - (1-np.transpose([y_c])) * np.log(1 - sigmoid(np.dot(X_c, theta_c))))
                               + (lambda_c/2/m_temp) * np.sum(theta_c[1:]**2))
    ones_c = np.array([X_c[:, 0]])
    grad_c[0] = (1/m_temp) * np.dot(ones_c, sigmoid(np.dot(X_c, theta_c)) - np.transpose([y_c]))
    grad_c[1:] = (1/m_temp) * np.dot(np.transpose(np.array(X_c[:, 1:])), sigmoid(np.dot(X_c, theta_c)) - np.transpose([y_c])) + (lambda_c/m_temp) * theta_c[1:]
    return J_c, grad_c


cost, grad = costFunctionReg(initial_theta, X, y, lambda_re)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print(grad[0:5])
print('Expected gradients (approx) - first five values only:')
print('0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

# Regularization and Accuracies
#
# this part is different from matlab code
# because in matlab, fminunc is used to calculated the min(cost)
# while I am still using gradient descent method here

iterations = 2500
alpha = 0.1
lambda_re = 1


def gradientDescent_reg(X_g, Y_g, theta_g, alpha_g, lambda_g, iterations_g):
    J_history = 0
    m_temp = len(Y_g)
    for i in range(iterations_g):
        theta_g[0] = theta_g[0] - alpha_g / m_temp * np.dot(np.transpose(X_g[:, 0]), np.dot(X_g, theta_g) - np.transpose([Y_g]))
        theta_g[1:] = theta_g[1:] * (1 - alpha_g * lambda_g / m_temp) - alpha_g / m_temp * np.dot(np.transpose(X_g[:, 1:]), np.dot(X_g, theta_g) - np.transpose([Y_g]))
        J_history, grad_g = costFunctionReg(theta_g, X_g, Y_g, lambda_g)

    return theta_g, J_history


theta_local, J_local = gradientDescent_reg(X, y, initial_theta, alpha, lambda_re, iterations)
print('theta new is', theta_local)
print('j new is', J_local)
# this is only a local optimal solution with initial guess all thetas equal to zero

# here is the best theta values from matlab fminunc
theta_test = [1.27246614841031, 0.624959476379526, 1.18098931154171, -2.01979932225575, -0.917387471999802, -1.43098663413990, 0.124272268903098, -0.365655539892576, -0.357260565690065, -0.175252176109096, -1.45795648589907, -0.0512086083634308, -0.615487542254826, -0.274806276466398, -1.19323169326991, -0.242246833039664, -0.206000870211619, -0.0448722295277553, -0.277888053691277, -0.295483805053953, -0.456065257485388, -1.04330043477467, 0.0276731548793733, -0.292457768268890, 0.0155141654649643, -0.327449549487439, -0.143816566566717, -0.924514980832185]
theta_test = np.transpose([theta_test])
J_best, g_best = costFunctionReg(theta_test, X, y, lambda_re)
print('Global optimal value J is:', J_best)

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((np.size(u), np.size(v)))
print('value:', np.shape(z))
print('value1:', np.shape(np.dot(mapFeature(u[1], v[1]), theta_test)))
print('z00 is', z[0][0])


def plotDecisionBoundary(theta_p):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((np.size(u), np.size(v)))

    for i in range(np.size(u)):
        for j in range(np.size(v)):
            z[i][j] = np.dot(mapFeature(u[i], v[j]), theta_p)

    z = z.T
    plt.contour(u, v, z, [0, 0.0000001])
    return


X = data[:, 0:-1]
y = data[:, 2]
plotscatter(X, y)
plotDecisionBoundary(theta_test)
plt.show()


