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
        Jn = [[f(xn) for f in A[r]] for r in range(len(xn))]
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

def compute_order(x,xstar):
# p_{n+1}-p (from the second index to the end)
  diff1 = np.linalg.norm(x[1::]-xstar)
  # p_n-p (from the first index to the second to last)
  diff2 = np.linalg.norm(x[0:-1]-xstar)
  # linear fit to log of differences
  fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
  print('the order equation is')
  print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
  print('lambda = ' + str(np.exp(fit[1])))
  print('alpha = ' + str(fit[0]))
  return [fit,diff1,diff2]

# ellipsoid
def E(x):
    return 16 - x[0]**2 - 4*x[1]**2 - 4*x[2]**2

def Ex(x):
    return -2*x[0]

def Ey(x):
    return -8*x[1]

def Ez(x):
    return -8*x[2]

def iterQ3(x0, f, fx, fy, fz, tol, nmax):
    x = [x0]
    for i in range(nmax):
        fx_x = fx(x0)
        fy_x = fy(x0)
        fz_x = fz(x0) 
        d = f(x0)/(fx_x**2 + fy_x**2 + fz_x**2)

        # apply iteration
        x1 = np.array([x0[0] - d*fx_x,
              x0[1] - d*fy_x,
              x0[2] - d*fz_x])
        x.append(x1)

        if(np.linalg.norm(x1 - np.array(x0)) < tol):
            return x1, x, i
        x0 = x1
    return x1, x, i

x0 = [1,1,1]
pt, appxs, i = iterQ3(x0, E, Ex, Ey, Ex, 1e-10, 100)

print("\n\n")
print("The initial guess", x0, " gives the approximation:", pt)
print("This took", i, "iterations")
print()

compute_order(appxs, pt)