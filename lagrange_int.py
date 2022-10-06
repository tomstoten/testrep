import mypkg.prini as prini
import mypkg.my2DPlot
import numpy as np
import numpy.linalg as la
import math

def driver():


    f = lambda x: math.exp(x)

    N = 3
    ''' interval'''
    a = 0
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = np.zeros(N+1)
    for jj in range(N+1):
        yint[jj] = f(xint[jj])
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval = np.zeros(Neval+1)
    for kk in range(Neval+1):
       yeval[kk] = eval_lagrange(xeval[kk],xint,yint,N)

    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
        
    plt = mypkg.my2DPlot(xeval,fex)
    plt.addPlot(xeval,yeval)
    plt.show()
         
    err = abs(yeval-fex)
    plt2 = mypkg.my2DPlot(xeval,err)
    plt2.show()            


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
  
    

       
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
