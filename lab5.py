import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mypkg.Iteration1D import Iteration1D


def newton(f,fp,p0,count,tol,Nmax):
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          p = p[:it+1]
          return [p,pstar,info,it+count]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it+count]

def new_newton(f, fprime, fpprime, a, b, tol, Nmax):
    count = 0

    fa = f(a); fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

    '''
     verify end point is not a root
    '''
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]

    gprime = lambda x: f(x) * fpprime(x) / (fprime(x)**2)

    while (count < Nmax):
      c = 0.5*(a+b)
      fc = f(c)

      if (abs(gprime(c)) < 1):
        # start newton iteration
        return newton(f, fprime, c, count, tol, Nmax)

      if (fa*fc<0):
         b = c
      elif (fb*fc<0):
        a = c
        fa = fc
      else:
        astar = c
        ier = 3
        return [astar, ier, count]

      if (abs(b-a)<tol):
        astar = a
        ier =0
        return [astar, ier, count]
      
      count = count +1

    astar = a
    ier = 2
    return [astar,ier,count]

f = lambda x: math.exp(x**2 + 7*x - 30) - 1
fp = lambda x: (2*x + 7) * math.exp(x**2 + 7*x - 30)
fpp = lambda x: (2 + (2*x + 7)**2) * math.exp(x**2 + 7*x - 30)

find = Iteration1D(f, 'newton')
find.a = 2; find.b = 4.5
find.fp = fp
find.p0 = 4.5
find.tol = 1e-13
find.Nmax = 100

appxs, root, ier, c = new_newton(f, fp, fpp, find.a, find.b, find.tol, find.Nmax)

print("For the interval:", find.a, "to", find.b, ", the root is x =", root, "ier:", find.info)
print("This took", c, "iterations")
