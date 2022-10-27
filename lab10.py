from argparse import ArgumentError
import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
from scipy import integrate as int

def driver():

#  function you want to approximate
    f = lambda x: 1.0/(1 + x**2)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.

# order of approximation
    n = 6

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
      pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
    
    # print(lookup)
    # create vector with exact values
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    fig, ax = plt.subplots(1,2)
    ax[0].plot(xeval,fex,color="gray", linestyle="dashed")
    ax[0].plot(xeval,pval,'-r')
    ax[0].set_title("Legendre Expansion")
    
    err = abs(pval-fex)
    ax[1].plot(xeval,err)
    ax[1].set_title("Error")
    plt.show()     

    
# lookup = {}
# def leg_helper(n):
#     if n in lookup:
#         return lookup[n]
#     else:
#         # base cases
#         if n < 0:
#             print("The argument n must be greater than or equal to 0")
#             return -1
#         if n == 0:
#             lookup[n] = lambda arg: 1
#             return 1
#         elif n == 1:
#             lookup[n-1] = lambda arg: 1
#             lookup[n] = lambda arg: arg
#             return lookup[n]
#         else:
#             phi_nm1 = leg_helper(n-2)
#             phi_n = leg_helper(n-1)
#             lookup[n] = lambda arg: 1/(n+1) * ((2*n + 1)*arg*phi_n - n*phi_nm1)
#             return lookup[n]
    

def eval_legendre(n, x):
    phi = np.zeros(n+1)
    phi[0] = 1
    if n > 0:
        phi[1] = x
        for j in range(1, n):
            phi[j+1] = 1/(j+1) * ((2*j + 1)*x*phi[j] - j*phi[j-1])
    return phi


    # leg_helper(n)
    # #print(lookup)
    # p = [lookup[i](x) for i in range(n+1)]
    # return p
    
def pj(j, x):
    p = eval_legendre(j, x)
    #print(p)
    return p[-1]

def eval_legendre_expansion(f,a,b,w,n,x): 

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
    p = eval_legendre(n, x)
    #print(p)

# initialize the evaluator with the constant term
    pval = 0.0   
    for j in range(0,n):
       ''' call your coefficient evaluator'''
       phi_j = lambda x: pj(j, x)
       psw = lambda x: phi_j(x)**2 * w(x)

       norm_factor, err = int.quad(psw, a, b)

       fpw = lambda x: phi_j(x) * f(x) * w(x) / norm_factor

       aj, err = int.quad(fpw, a, b)
       pval = pval + aj*p[j] 
       
    return(pval)
    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()               
