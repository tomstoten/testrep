import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random

xeval = np.linspace(0,10,1000)
xint = np.linspace(0,10,11)

def pointsInInterval(arr, int):
    ind1 = np.where(arr >= int[0])[0]
    ind2 = np.where(arr < int[1])[0]
    ind = [x for x in ind1 if x in ind2]
    return ind

def getPoints(xeval,xint,i):
    return pointsInInterval(xeval, [xint[i], xint[i+1]])

def getLine(p1, p2):
    # get slope
    m = (p2[1] - p1[1])/(p2[0] - p1[0])
    # point-slope form
    f = lambda x: m*(x-p1[0]) + p1[1]
    return f

def driver():
    f = lambda x: math.exp(x)
    f = lambda x: 1./(1 + (10*x)**2)
    a = -1
    b = 1
    ''' create points you want to evaluate at'''

    Neval = 100
    xeval = np.linspace(a,b,Neval)
    ''' number of intervals'''
    Nint = 10
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    #print(yeval)

    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)

    for j in range(Neval):
        fex[j] = f(xeval[j])
    
    fig, ax = plt.subplots(1,2)
    ax[0].plot(xeval,fex,'r-')
    ax[0].plot(xeval,yeval,'b-')

    err = abs(yeval-fex)
    ax[1].plot(xeval,err)

    plt.show()

def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        ind = getPoints(xeval, xint, jint)
        vals = np.take(xeval,ind)
        #print(ind)

        '''let ind denote the indices in the intervals'''
        
        '''let n denote the length of ind'''
        n = len(ind)
        
        '''temporarily store your info for creating a line in the interval of
        interest'''
        
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        for kk in range(n):
            '''use your line evaluator to evaluate the lines at each of the points
            2
            in the interval'''
            line = getLine([a1, fa1], [b1, fb1])
            '''yeval(ind(kk)) = call your line evaluator at xeval[ind[kk]] with
            the points (a1,fa1) and (b1,fb1)'''
            yeval[ind[kk]] = line(vals[kk])

    return yeval

if __name__ == '__main__':
    driver()