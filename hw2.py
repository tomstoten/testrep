import math
import numpy as np
#from prini import prini as prin
import matplotlib.pyplot as plt
import random

def func(x):
    y = math.e ** x
    return y - 1

def taylor(x):
    sum = 0
    for i in range(1, 17):
        sum += (x**i)/fact(i)
        print(sum)
    return sum

def fact(n):
    if n == 1:
        return 1
    else:
        return n * fact(n-1)

def sum(n, t, y):
    total = 0
    for k in range(1, n):
        total += t[k] * y[k]
    return total

def x(theta, R=1.2, dr=0.1, f=15, p=0):
    return R*(1 + dr*np.sin(f*theta + p))*np.cos(theta)

def y(theta, R=1.2, dr=0.1, f=15, p=0):
    return R*(1 + dr*np.sin(f*theta + p))*np.sin(theta)


if __name__ == '__main__':
    #x = 9.999999995e-10
    #x = 1e-9
    #dx = 1e-6
    #print(func(x + dx) - func(x))
    #p = np.expm1(x)
    #ps = func(x)
    #print(taylor(x))
    #print(abs(p - ps))
    #print(abs(p-ps)/p)

    t = np.linspace(0, np.pi, 31)
    y_ = np.cos(t)

    
    #pr = prin("real", "the sum is:", sum(31, t, y_))
    #pr.print()


    thetas = np.linspace(0, 2*np.pi, 200)
    
    fig, ax = plt.subplots()
    #xv = x(thetas)
    #yv = y(thetas)
    
    for i in range(10):
        xv = x(thetas, R=i, dr=0.05, f=2+i, p=random.uniform(0, 2))
        yv = y(thetas, R=i, dr=0.05, f=2+i, p=random.uniform(0, 2))
        ax.plot(xv, yv)
    #ax.plot(xv, yv)
    #ax.set(xlim=(-1, 2), ylim=(-1, 2))
    #plt.savefig('hw2_p5(2).pdf')
    plt.show()
    
    #print(t)