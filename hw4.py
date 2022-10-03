import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mypkg.Iteration1D import Iteration1D

def T(x, t=5184000, alpha=1.38e-7, T_i=20, T_s=-15):
    return math.erf(x / (2 * math.sqrt(alpha * t) ) ) * (T_i - T_s) + T_s

def Tprime(x, t=5184000, alpha=1.38e-7, T_i=20, T_s=-15):
    return 23.24646163 * math.exp(-0.34945876 * x**2)

def func(x):
    return x**6 - x - 1

def fprime(x):
    return 6 * x**5 - 1

f = lambda x: func(x)
fp = lambda x: fprime(x)
find = Iteration1D(f, 'secant')
find.a = 2; find.b = 1
find.fp = fp
find.p0 = 2
find.tol = 1e-13
find.Nmax = 100

appxs, root, ier, c = find.root()

#print("For the interval:", find.a, "to", find.b, ", the root is x =", root, "ier:", find.info)
print("For the initial guess:", find.p0, ", the root is x =", root, "ier:", find.info)
print("This took", c, "iterations")

errs = [abs(g - root) for g in appxs]
print("Approximation | Abs Err |\n")
for i in range(len(errs)):
    print(appxs[i], errs[i])

num = errs[1:]
den = errs[:-1]

fig, ax = plt.subplots()
plt.loglog(den, num)
ax.set_title("Fig. 2: Secant method")
ax.set_xlabel(r"|x_{k} - a|")
ax.set_ylabel(r"|x_{k+1} - a|")
ax.axhline(y=0, color='black')
ax.axvline(x=0, color='black')

plt.grid()
plt.savefig("hw4_p5(2).pdf")
plt.show()
'''
xvec = np.linspace(0, 1, 10000)
yvec = []
for x in xvec:
    yvec.append(T(x))

fig, ax = plt.subplots()
ax.plot(xvec, yvec)
ax.set_title("Fig. 1")
ax.set_xlabel("x")
ax.set_ylabel("T(x,t=5184000) = f(x)")
ax.axhline(y=0, color='black')
ax.axvline(x=0, color='black')

plt.grid()
plt.savefig("hw4_p1(1).pdf")
plt.show()
'''