import testrep.mypkg.my2DPlot as myplt
import matplotlib.pyplot as plt
import numpy as np

def p(x):
    return (x-2)**9

def p_exp(x):
    return x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512

def f(x, delta):
    return np.cos(x + delta) - np.cos(x)

def f_(x, delta):
    return -2 * np.sin(2*x + delta) * np.sin(delta/2)
    return (np.cos(x + delta)**2 + np.sin(x)**2 - 1)/(np.cos(x + delta) + np.cos(x))

def f_star(x, delta):
    return -1 * delta * np.sin(x) - delta**2 * np.cos(x + delta/2) / 2


xvec = np.logspace(-16, 0, num=17)
x = 3.1415926535
x = 1e6
plt.plot(xvec, f(x, xvec))
#plt.plot(xvec, f_star(x, xvec))
plt.xlabel('delta (x = 1e6)')
plt.ylabel('f(x, delta) and f**(x, delta)')
plt.xscale('log')
#plt.savefig('hw1_p5(4).pdf')
plt.show()
'''
plt = myplt(lambda x: p_exp(x), 1.920, 2.080)
plt.labels('x', 'p(x)')
plt.addPlot(lambda x: p(x))
plt.color('black')
plt.dotted()
plt.save('hw1_p1.pdf')
plt.show()
'''
