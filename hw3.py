import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mypkg.Iteration1D import Iteration1D

#f = lambda x: np.sin(x) - 2*x + 1
#f = lambda x: (x-5)**9
#f = lambda x: x**9 - 45*x**8 + 900*x**7 - 10500*x**6 + 78750*x**5 - 393750*x**4 + 1312500*x**3 - 2812500*x**2 + 3515625*x - 1953125
#f = lambda x: x**3 + x - 4
f = lambda x: -np.sin(2*x) + 5*x/4 - 3/4

find = Iteration1D(f, "fixedpt")
find.tol = 1e-10
find.Nmax = 100

#find.a = -1; find.b = 0
find.p0 = 3
x, c = find.root()
print("For p0:", find.p0, ", the root is x =", x, find.info)
print("This took", c, "iterations")

def f_(x):
    return x - 4*np.sin(2*x) - 3


xvec = np.linspace(-2, 7.5, 10000)
yvec = f_(xvec)

fig, ax = plt.subplots()
ax.plot(xvec, yvec)
ax.axhline(y=0, color='black')
ax.axvline(x=0, color='black')

#plt.savefig("hw3_p5(1).pdf")
#plt.show()