import math
import numpy as np
import matplotlib.pyplot as plt
import random

f = lambda x: 1.0/(1. + (10.*x)**2)

N = 100
X = np.linspace(-1, 1, N+1)

Neval = 1000
#Xeval = np.array([-1 + (j-1) * 2/(Neval-1) for j in range(Neval)])
Xeval = np.array([np.cos((2*j - 1)*np.pi/(2*(N-1))) for j in range(1,N)])
#print(Xeval)
'''
V = np.zeros((N+1, N+1))
for j in range(N+1):
    V[:,j] = X**j

#print("The Vandermonde matrix is:\n")
#print(V)

Veval = np.zeros((Neval, N+1))
for i in range(N+1):
    Veval[:,i] = Xeval**i
'''
F = f(X)

Feval = f(Xeval)

# Vc = F
#c = np.linalg.solve(V, F)

# fig,ax = plt.subplots(1,2)
# ax[0].plot(Xeval, Feval, '-')
# ax[0].plot(Xeval,Veval@c, 'b-')
# ax[1].plot(Xeval, np.abs(Feval - Veval@c))
# plt.show()

def driver():
    xeval = np.linspace(-1,1,Neval)
    yevalL = np.zeros(Neval)
    for kk in range(Neval):
       yevalL[kk] = eval_lagrange(xeval[kk],X,F,N)

    fineGrid = 1000
    xvec = np.linspace(-1, 1, fineGrid)

    ''' create vector with exact values'''
    fex = np.zeros(fineGrid)
    for kk in range(fineGrid):
        fex[kk] = f(xvec[kk])

    fx = np.zeros(Neval)
    for kk in range(Neval):
        fx[kk] = f(xeval[kk])

    yevalB = np.zeros(Neval)
    for i in range(Neval):
        yevalB[i] = barycentric(xeval[i],Xeval,F,N-2)

    #print(yevalB)
        
    fig,ax = plt.subplots(1,2)

    ax[0].plot(xvec,fex,'b-')
    ax[0].plot(xeval,yevalB,'r-')
    ax[0].set_ylabel("f(x)")
    ax[0].set_xlabel("x")
    ax[0].set_title("Barycentric Approximation of f(x) N=100")
         
    errL = abs(yevalB-fx)
    ax[1].plot(xeval,errL,'r-')
    #ax[1].plot(Xeval, np.abs(Feval - Veval@c),'b-')
    ax[1].set_title("Error")
    plt.savefig("hw7_2.pdf")
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

def barycentric(x,xint,yint,N):
    def l(x,n):
        ans = 1
        for k in range(n+1):
            ans *= (x-xint[k])
        return ans
    def w(n,j):
        ans = 1
        for k in range(n+1):
            if k != j:
                ans *= (xint[j] - xint[k])
        return 1.0/ans
    p = lambda x: l(x,N) * sum([w(N,j) * yint[j]/(x - xint[j]) for j in range(N+1)])
    return p(x)

       
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()