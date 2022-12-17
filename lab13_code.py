import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from timeit import default_timer as timer


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     ns = [100, 500, 1000, 2000, 4000, 5000]
     for N in ns:
          print("N =",N)
 
          ''' Right hand side'''
          b = np.random.rand(N,10)
          A = np.random.rand(N,N)

          t0 = timer()
          for i in range(10):
               x = scila.solve(A,b[:,i])
          
          t1 = timer()

          dt = t1-t0
          
          print("Solving 10 eq. without LU took",dt,"seconds to compute")
          

          # ''' Create an ill-conditioned rectangular matrix '''
          # N = 10
          # M = 5
          # A = create_rect(N,M)     
          # b = np.random.rand(N,1)
          
          # Factoring LU
          t0 = timer()
          lu, p = scila.lu_factor(A)

          t1 = timer()

          dt = t1-t0
          print("Computing LU took",dt,"seconds to compute")

          t0 = timer()
          for i in range(10):
               x = scila.lu_solve((lu, p), b[:,i])

          t1 = timer()

          dt = t1-t0

          print("Solving 10 eq. with LU took",dt,"seconds to compute\n")
          print("*********************")
     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
