import mypkg.prini as prini
import mypkg.my2DPlotB
import numpy as np
import numpy.linalg as la
import math
import scipy.integrate.quad as quad

def driver():

#  function you want to approximate
    f = lambda x: math.exp(x)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
      pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
        
    plt = mypkg.my2DPlotB(xeval,fex)
    plt.addPlot(xeval,pval)
    plt.show()
    
    err = abs(pval-fex)
    plt2 = mypkg.my2DPlotB(xeval,err)
    plt2.logy()
    plt2.show()            
    
      
    

def eval_legendre_expansion(f,a,b,w,n,x): 

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
    p = ...

# initialize the evaluator with the constant term
    pval = p[0]    
    for j in range(1,n):
       ''' call your coefficient evalutor'''             
       aj = ...
       pval = pval +aj*p[j] 
       
    return(pval)
    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()               
