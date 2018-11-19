
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("C://Users/Stacy/Documents/Eric's Courses/Andrew Ng's Machine Learning/Exercise 1 Part 2/")

data = pd.read_csv('ex1data1.txt', header = None)

x = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)

#plt.scatter(x,y)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s')
#plt.show()

x = x[: , np.newaxis]
y = y[:, np.newaxis]

theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
x = np.hstack((ones,x))

def computeCost(x,y, theta):
    temp = np.dot(x, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

J = computeCost(x, y, theta)
#print (J)

def gradientDescent (x, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(x, theta) - y
        temp = np.dot(x.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescent(x, y, theta, alpha, iterations)

print(theta)

J = computeCost(x, y, theta)
print(J)


















