import math
import numpy as np
import matplotlib.pyplot as plt
import random

f = lambda x: 1.0/(1. + (10.*x)**2)

N = 5
X = np.linspace(-1, 1, N+1)

Neval = 1000
Xeval = np.array([-1 + (j-1) * 2/(Neval-1) for j in range(Neval)])

V = np.zeros((N+1, N+1))
for j in range(N+1):
    V[:,j] = X**j


Veval = np.zeros((Neval, N+1))
for i in range(N+1):
    Veval[:,i] = Xeval**i

F = f(X)

Feval = f(Xeval)

# Vc = F
c = np.linalg.solve(V, F)

# fig,ax = plt.subplots(1,2)
# ax[0].plot(Xeval, Feval, '-')
# ax[0].plot(Xeval,Veval@c, 'b-')
# ax[1].plot(Xeval, np.abs(Feval - Veval@c))
# plt.show()

def driver():
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = F[j]

    y = dividedDiffTable(X, y, N+1)

    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(-1,1,Neval+1)
    yevalL = np.zeros(Neval+1)
    yevalNDD = np.zeros(Neval+1)
    for kk in range(Neval+1):
       yevalL[kk] = eval_lagrange(xeval[kk],X,F,N)
       yevalNDD[kk] = evalDDpoly(xeval[kk],X,y,N)


    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
        
    fig,ax = plt.subplots(1,2)
    ax[0].plot(xeval,fex,'b-')
    ax[0].plot(xeval,yevalL,'r-')
    ax[0].plot(xeval,yevalNDD,'y-')
         
    errL = abs(yevalL-fex)
    errNDD = abs(yevalNDD-fex)
    ax[1].plot(xeval,errL,'r-')
    ax[1].plot(xeval,errNDD,'y-')
    ax[1].plot(Xeval, np.abs(Feval - Veval@c),'b-')
    plt.show()


def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)          

''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return y
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

       
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()