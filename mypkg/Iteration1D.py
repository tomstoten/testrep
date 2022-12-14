import numpy as np

class Iteration1D:

    def __init__(self,f,method):

      self.f = f
      self.method = method
      
      # initial interval for bisection
      self.a = None
      self.b = None
      # initial guess for newton/fixpt
      self.p0 = None
      self.fp = None
      # tolerances and max iter
      self.tol = None
      self.Nmax = None
      # info message
      self.info = None
      # root
      self.pstar = None
      # iters for newton
      self.p_iters = None
      # final iteration count
      self.count = None
      self.appxs = None

    def root(self):

      if self.method == 'bisection':
        if self.a is None or self.b is None or \
           self.tol is None or self.Nmax is None:
          print('error: some attributes for bisection aresys. returning ..')
          return
        [self.pstar, self.info, self.count] = bisection(self.f,self.a,self.b,self.tol,self.Nmax)

      elif self.method == 'newton':
        if self.fp is None or self.p0 is None or \
           self.tol is None or self.Nmax is None:
          print('error: some attributes for newton are unset. returning ..')
          return

        [self.p_iters, self.pstar, self.info, self.count] = \
          newton(self.f,self.fp,self.p0,self.tol,self.Nmax)
      elif self.method == 'secant':
        if self.a is None or self.b is None or self.p0 is None or \
           self.tol is None or self.Nmax is None:
          print('error: some attributes for newton are unset. returning ..')
          return

        [self.p_iters, self.pstar, self.info, self.count] = \
          secant(self.f,self.a,self.b,self.tol,self.Nmax)

      elif self.method == 'fixedpt':
        if self.p0 is None or self.tol is None or self.Nmax is None:
          print('error: some attributes for fixedpt are unset. returning ..')
          return 
        [self.pstar, self.appxs, self.info, self.count] = fixedpt(self.f,self.p0,self.tol,self.Nmax)
      
      return self.p_iters, self.pstar, self.info, self.count


def secant(f, a, b, tol, Nmax):
  p = np.zeros(Nmax+1)
  p[0] = a
  p[1] = b
  for i in range(1, Nmax):
    pn = p[i]
    pm1 = p[i-1]
    pn1 = pm1 - (f(pm1) * (pn-pm1) / (f(pn) - f(pm1)))
    p[i+1] = pn1
    if abs(pn1 - pn) < tol:
      pstar = pn1
      ier = 0
      p = p[:i+2]
      return [p, pstar, ier, i]
  pstar = pn1
  info = 1
  return [p,pstar,info,i]


def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          p = p[:it+1]
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]

def bisection(f,a,b,tol,Nmax):
    """
    Inputs:
      f,a,b       - function and endpoints of initial interval
      tol, Nmax   - bisection stops when interval length < tol
                  - or if Nmax iterations have occured
    Returns:
      astar - approximation of root
      ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    """

    '''
     first verify there is a root we can find in the interval
    '''
    count = 0

    fa = f(a); fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

    '''
     verify end point is not a root
    '''
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]

    while (count < Nmax):
      c = 0.5*(a+b)
      fc = f(c)

      if (fc ==0):
        astar = c
        ier = 0
        return [astar, ier, count]

      if (fa*fc<0):
         b = c
      elif (fb*fc<0):
        a = c
        fa = fc
      else:
        astar = c
        ier = 3
        return [astar, ier, count]

      if (abs(b-a)<tol):
        astar = a
        ier =0
        return [astar, ier, count]
      
      count = count +1

    astar = a
    ier = 2
    return [astar,ier,count] 


def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0

    x = np.zeros((Nmax,1))

    x[0] = x0
    while (count <Nmax):
      count = count +1
      x1 = f(x0)
      x[count] = x1
      if (abs(x1-x0)/abs(x1) < 0.5*tol):  # relative error now
        xstar = x1
        ier = 0
        x = x[0:count]
        return [xstar,x,ier,count]
      x0 = x1

    xstar = x1
    ier = 1
    x = x[0:count]
    return [xstar, x, ier, count]