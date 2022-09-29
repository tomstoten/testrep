import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mypkg.Iteration1D import Iteration1D

# prelab

def f(x):
    return np.cos(x)

h = 0.01 * 2.**(-np.arange(0,10))
# print("h:", h)

def fp_fw(s, h):
    return (f(s + h) - f(s)) / h

def fp_ct(s, h):
    return (f(s + h) - f(s - h)) / (2*h)

def compute_order(x,xstar):
# p_{n+1}-p (from the second index to the end)
  diff1 = np.abs(x[1::]-xstar)
  # p_n-p (from the first index to the second to last)
  diff2 = np.abs(x[0:-1]-xstar)
  # linear fit to log of differences
  fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
  print('the order equation is')
  print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
  print('lambda = ' + str(np.exp(fit[1])))
  print('alpha = ' + str(fit[0]))
  return [fit,diff1,diff2]

# f'(pi/2) = -sin(pi/2) = -1
x = np.pi / 2
appxs_fw = fp_fw(x, h)
appxs_ct = fp_ct(x, h)

fp_x = -np.sin(x)

print("The forward approximations are:", appxs_fw)
print("The centered approximations are:", appxs_ct)
print()

fit_fw, diff1_fw, diff2_fw = compute_order(appxs_fw, fp_x)
fit_ct, diff1_ct, diff2_ct = compute_order(appxs_ct, fp_x)


# lab exercises

F = [
    lambda x: 4*x[0]**2 + x[1]**2 - 4,
    lambda x: x[0] + x[1] - np.sin(x[0] - x[1]) 
]

A = [[lambda x: 8*x[0],                  lambda x: 2*x[1]                 ],
     [lambda x: 1 - np.cos(x[0] - x[1]), lambda x: 1 + np.cos(x[0] - x[1])]]

x0 = [1, 0]

tol = 1e-10
nmax = 100

def slackerNewton(F, xn, A, tol, nmax):
    jacob_comps = 1
    # compute the first inverse Jacobian
    J0 = [[f(xn) for f in A[r]] for r in range(len(xn))]
    J_i = np.linalg.inv(J0)
    for i in range(nmax):

        # Apply Newton formula
        F_of_xn = [f(xn) for f in F]
        
        xn1 = xn - np.matmul(J_i, [f(xn) for f in F])

        F_of_xn1 = [f(xn1) for f in F]

        # check if converging
        if(np.linalg.norm(xn1 - xn) < tol):
            return xn1, i, jacob_comps

        # if |F(x_{n+1})| > |F(x_{n})|, recompute the Jacobian
        if(np.linalg.norm(F_of_xn) < 100*np.linalg.norm(F_of_xn1)):
            # recalculate inverse Jacobian
            Jn = [[f(xn) for f in A[r]] for r in range(len(xn))]
            J_i = np.linalg.inv(Jn)
            jacob_comps += 1
        
        xn = xn1
    return xn1, i, jacob_comps

xn, count, comps = slackerNewton(F, x0, A, tol, nmax)
print("The approximation is:", xn,"\nThis took", count, "iterations and", comps,"Jacobian computations")
