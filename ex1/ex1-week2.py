# %% Machine Learning Online Class - Exercise 1: Linear Regression
#
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  linear exercise. You will need to complete the following functions
# %  in this exericse:
# %
# %     warmUpExercise.m
# %     plotData.m
# %     gradientDescent.m
# %     computeCost.m
# %     gradientDescentMulti.m
# %     computeCostMulti.m
# %     featureNormalize.m
# %     normalEqn.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
# % x refers to the population size in 10,000s
# % y refers to the profit in $10,000s
#
# Instructions copied from matlab code
# For reference and personal use only
# I am a new python learner
# Discussion and suggestions are welcome

import numpy as np
import warmUpExercise as wue
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

# def warmUpExercise():
#     A = numpy.identity(5)
#     return A
# to make it clearly, write these def functions into another .py file
# try to import the function from another .py file
A = wue.get_matrix()
print(A)

# input("Press the <ENTER> key to continue...")
# change this to a pause function to make it easier to code, maybe write it in wue
wue.pause()

print('Plotting Data ...\n')

# one way to read data, while I currently have no idea how to deal easily with it in matrix form
# with open('ex1data1.txt', 'r') as ex1data1:
    # lines = ex1data1.readlines()
    # X = []
    # Y = []
    # for line in lines:
    #     lineData = line.strip().split(',')
    #     X.append(float(lineData[0]))
    #     Y.append(float(lineData[1]))
    # m = len(X)

# another way to read data using numpy
data = np.loadtxt('ex1data1.txt', dtype=np.float, delimiter=',')
X_original = data[:,0]
Y_original = data[:,1]
X = X_original
Y = Y_original
m = len(X)
pl.plot(X,Y,'rx')
pl.title('machine learning example')
pl.xlabel('population in 1000s')
pl.ylabel('profit in $10,000s')
pl.show()

wue.pause()
X = np.transpose(np.array([X]))  # 2D array should have the extra [] outside the 1D array
Y = np.transpose(np.array([Y]))
ones = np.ones(m)
ones = np.transpose(np.array([ones]))

X = np.append(ones, X, axis=1)
theta = np.zeros((2,1))

iterations = 1500
alpha = 0.01

print('Testing the cost function ...')


def computeCost(X, Y, theta):
    m_temp = len(Y)
    temp1 = np.matmul(X, theta) - Y
    temp1 = np.squeeze(np.asarray(temp1))
    temp2 = np.dot(temp1, temp1)
    J = np.sum(temp2) / (2 * m_temp)
    return J


J = computeCost(X, Y, theta)
print('With theta = [0 ; 0]\nCost computed =', J)
J = computeCost(X, Y, [[-1], [2]])
print('With theta = [-1 ; 2]\nCost computed =', J)

wue.pause()
print('Running Gradient Descent ...')


def gradientDescent(X, Y, theta, alpha, iterations):
    J_history = 0;
    m_temp = len(Y)
    for i in range(iterations):
        theta = theta - alpha / m_temp * np.matmul(np.transpose(X), np.matmul(X, theta)-Y)
        J_history = computeCost(X, Y, theta)

    return theta, J_history


theta, J = gradientDescent(X, Y, theta, alpha, iterations)

# not sure how to draw another graph on the first graph here using matplotlib
# so I have to close the first one, and redraw this one
pl.plot(X_original, Y_original, 'rx', label='Training data')
pl.title('machine learning example')
pl.xlabel('population in 1000s')
pl.ylabel('profit in $10,000s')
pl.plot(X[:, 1], np.matmul(X, theta), 'b', label='Linear regression')
pl.legend()
pl.show()

predict1 = np.matmul([1, 3.5], theta)
print('For population = 35,000, we predict a profit of', predict1*10000)
predict2 = np.matmul([1, 7], theta)
print('For population = 35,000, we predict a profit of', predict2*10000)
wue.pause()

fig = pl.figure()
ax = Axes3D(fig)

theta0_vals = np.array([np.arange(-10, 10, 20/99)])
theta1_vals = np.array([np.arange(-1, 4, 5/99)])

J_vals = np.zeros((np.size(theta0_vals), np.size(theta1_vals)))
for i in range(np.size(theta0_vals)):
    for j in range(np.size(theta1_vals)):
        t = [[theta0_vals[0, i]], [theta1_vals[0, j]]]
        J_vals[[i], [j]] = computeCost(X, Y, t)


J_vals = np.transpose(J_vals)
theta1_vals, theta0_vals = np.meshgrid(theta1_vals, theta0_vals)
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
cset = ax.contour(theta0_vals, theta1_vals, J_vals, zdir='z', offset=-np.pi, cmap=cm.coolwarm)  # show concur plot on the bottom of 3D plot
ax.set_zlim(0, 800)
fig.colorbar(surf, shrink=0.5, aspect=5)
pl.show()
wue.pause()

pl.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, num=20))
pl.show()
