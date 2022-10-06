import mypkg.prini as prini
import mypkg.my2DPlot
import numpy as np
import numpy.linalg as la
import math

def driver():

    f = lambda x: math.exp(x)

    N = 10
    ''' interval'''
    a = 0
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = np.zeros(N+1)
    for jj in range(N+1):
        yint[jj] = f(xint[jj])

    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval = np.zeros(Neval+1)
    for kk in range(Neval+1):
        yeval[kk] = evalDDpoly(xeval[kk],xint,y,N)

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


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
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
