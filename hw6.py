import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mypkg.Iteration1D import Iteration1D

# Problem 1
A1 = [[lambda x: 2*x[0],         lambda x: 2*x[1]],
      [lambda x: math.exp(x[0]), lambda x: 1     ]]

F1 = [lambda x: x[0]**2 + x[1]**2 - 4,
      lambda x: math.exp(x[0]) + x[1] - 1]

g1 = np.array([1, 1])
g2 = np.array([1, -1])
g3 = np.array([0, 0])

tol = 1e-10
nmax = 100

def newton2D(F, xn, A, tol, nmax):
    for i in range(nmax):
        Jn = [[f(xn) for f in A[r]] for r in range(len(xn))]
        Jn_i = np.linalg.inv(Jn)
        xn1 = xn - np.matmul(Jn_i, [f(xn) for f in F])
        #print(xn1)
        if(np.linalg.norm(xn1 - xn) < tol):
            return xn1, i
        xn = xn1
    return xn1, i

def Broyden(F, xn, A, tol, nmax):
    # compute the first inverse Jacobian
    Jn = [[f(xn) for f in A[r]] for r in range(len(xn))]
    # take first inverse directly
    J_i = np.linalg.inv(Jn)
    for i in range(nmax):
        # Apply Newton formula
        xn1 = xn - np.matmul(J_i, [f(xn) for f in F])

        # check if converging
        if(np.linalg.norm(xn1 - xn) < tol):
            return xn1, i

        # recompute inverse Jacobian using Broyden method
        F_of_xn = np.array([f(xn) for f in F])
        F_of_xn1 = np.array([f(xn1) for f in F])

        diff = np.array(xn1 - xn)

        x = F_of_xn1 - F_of_xn - np.matmul(Jn, diff)
        x = np.array(x) / (np.linalg.norm(diff) ** 2)

        # numpy stuff to make sure arrays are correct shape
        x = np.array([x]).T
        y = np.array([diff])

        Ax = np.matmul(J_i, x)
        Axy = np.matmul(Ax, y)

        J_i = J_i - np.matmul(Axy, J_i) / (1 + np.matmul(np.matmul(y, J_i), x))

        xn = xn1
    return xn, i     

def slackerNewton(F, xn, A, tol, nmax):
    jacob_comps = 1
    # compute the first inverse Jacobian
    J0 = [[f(xn) for f in A[r]] for r in range(len(xn))]
    J_i = np.linalg.inv(J0)
    for i in range(nmax):
        # Apply Newton formula
        xn1 = xn - np.matmul(J_i, [f(xn) for f in F])
        #print(xn1)
        # check if converging
        if(np.linalg.norm(xn1 - xn) < tol):
            return xn1, i, jacob_comps

        # randomly recompute Jacobian
        if(np.random.random() < 1):
            # recalculate inverse Jacobian
            Jn = [[f(xn1) for f in A[r]] for r in range(len(xn1))]
            J_i = np.linalg.inv(Jn)
            jacob_comps += 1
        
        xn = xn1
        # print(xn, J_i)
    return xn1, i, jacob_comps


xnn, countn = newton2D(F1, g1, A1, tol, nmax)
print("The Newton appx is:", xnn,"\nThis took", countn, "iterations")

xn, count, comps = slackerNewton(F1, g1, A1, tol, nmax)
print("The Slacker appx is:", xn,"\nThis took", count, "iterations and", comps,"Jacobian computations")

xnb, countb = Broyden(F1, g2, A1, tol, nmax)
print("The Broyden appx is:", xnb,"\nThis took", countb, "iterations")


# Problem 2

def steepestD():
    return

F2 = [lambda x: x[0] + np.cos(x[0]*x[1]*x[2]) - 1,
      lambda x: (1-x[0])**0.25 + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1,
      lambda x: -x[0]**2 - 0.1*x[1]**2 + 0.001*x[1] + x[2] - 1
]

A2 = [[lambda x: 1 - x[1]*x[2]*np.sin(x[0]*x[1]*x[2]), lambda x: -x[0]*x[2]*np.sin(x[0]*x[1]*x[2]), lambda x: - x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
[lambda x: -0.25*(1-x[0])**(-0.75), lambda x: 1, lambda x: 0.1*x[2] - 0.15],[
lambda x: -2*x[0], lambda x: -0.2*x[1] + 0.01, lambda x: 1]
]

tol = 1e-6

x0 = np.array([1, -1, 1])
xn2, i2 = newton2D(F2, x0, A2, tol, nmax)
print("The Newton appx is:", xn2,"\nThis took", i2, "iterations")