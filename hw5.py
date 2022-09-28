import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mypkg.Iteration1D import Iteration1D


def iterate(F, xn, A, tol, nmax):
    for i in range(nmax):
        xn1 = xn - np.matmul(A, [f(xn) for f in F])
        if(np.linalg.norm(xn1 - xn) < tol):
            return xn1, i
        xn = xn1
    return xn1, i

def newton2D(F, xn, A, tol, nmax):
    for i in range(nmax):
        Jn = [[f(xn) for f in A[0]],
              [f(xn) for f in A[1]]]
        Jn_i = np.linalg.inv(Jn)
        xn1 = xn - np.matmul(Jn_i, [f(xn) for f in F])
        if(np.linalg.norm(xn1 - xn) < tol):
            return xn1, i
        xn = xn1
    return xn1, i


F1 = [
    lambda x: 3*x[0]**2 - x[1]**2,
    lambda x: 3*x[0]*x[1]**2 - x[0]**3 - 1
]

x0 = [1,1]

A0 = [[1/6, 1/18],
      [0  , 1/6 ]]

xn, count = iterate(F1, x0, A0, 1e-13, 100)
print(xn, count)
print([f(xn) for f in F1])


A = [[lambda x: 6*x[0],                lambda x: -2*x[1]    ],
     [lambda x: 3*x[1]**2 - 3*x[0]**2, lambda x: 6*x[0]*x[1]]]
     
xn, count = newton2D(F1, x0, A, 1e-13, 100)
print(xn, count)
print([f(xn) for f in F1])

'''
F = [
    lambda x: 5*x[0] - 3,               #f1(x) 
    lambda x: x[0]**2 - 4,              #f2(x)
    lambda x: 3*x[0]**3 - 4*x[0] + 1    #f3(x)
    ]

x = [0.1, 0.1, -0.1]
F_of_x = [f(x) for f in F]
print(F_of_x)
'''
