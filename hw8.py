import math
import numpy as np
import matplotlib.pyplot as plt
import random

def eval_lagrange(xeval,xint,yint,N):
    lj = np.ones(N)
    
    for count in range(N):
       for jj in range(N):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)


def hermite(xeval,xint,yint,yintp,N):
    def l(n,j,x):
        ans = 1
        for i in range(n+1):
            if i != j:
                ans *= (x - xint[i])/(xint[j] - xint[i])
        return ans
    
    def lp_helper(n, j, i, x):
        ans = 1
        for m in range(n+1):
            if m != j and m != i:
                ans *= (x - xint[m])/(xint[j] - xint[m])
        return ans

    def lp(n,j):
        ans = 0
        for i in range(n+1):
            if i != j:
                ans += 1/(xint[j] - xint[i]) * lp_helper(n, j, i, xint[j])            
        return ans

    def Q(j,x):
        return (1 - 2*(x - xint[j]) * lp(N,j)) * l(N,j,x)**2
    
    def R(j,x):
        return (x - xint[j]) * l(N, j, x)**2
    
    yeval = 0
    for j in range(N):
        yeval += yint[j] * Q(j, xeval) + yintp[j] * R(j, xeval)
    
    return yeval



def pointsInInterval(arr, int):
    ind1 = np.where(arr >= int[0])[0]
    ind2 = np.where(arr <= int[1])[0]
    ind = [x for x in ind1 if x in ind2]
    return ind

def getPoints(xeval,xint,i):
    return pointsInInterval(xeval, [xint[i], xint[i+1]])

def getCubic(x1, fx1, x2, fx2, m1, m2, x):
    h = x2 - x1
    c = fx1 / h - m1*h / 6
    d = fx2 / h - m2*h / 6

    x2mx = x2 - x
    xmx1 = x - x1

    return (m1 * (x2mx)**3) / (6*h) + (m2 * (xmx1)**3) / (6*h) + c*(x2mx) + d*(xmx1)

def eval_cub_spline(xeval,Neval,a,b,f,Nint,clamp=True):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)

    A = np.zeros((Nint+1,Nint+1))
    q = np.zeros(Nint+1)

    h = []
    fofx = []
    for jint in range(Nint+1):
        a1= xint[jint]
        fa1 = f(a1)
        fofx.append(fa1)
        if jint < Nint:
            b1 = xint[jint+1]
            fb1 = f(b1)

        h.append(b1 - a1)

        row = np.zeros(Nint+1)
        if jint == 0 or jint == Nint:
            # natural boundary
            row[jint] = 1
            A[jint] = row
            q[jint] = 0
        else:
            row[jint - 1] = h[jint - 1] / 6
            row[jint] = (h[jint - 1] + h[jint]) / 3
            row[jint + 1] = h[jint] / 6
            A[jint] = row

            q[jint] = (fb1 - fa1) / h[jint] - (fa1 - fofx[jint - 1]) / h[jint - 1]
    
    M = np.linalg.solve(A, q)

    for j in range(Nint):
        ind = getPoints(xeval, xint, j)
        vals = np.take(xeval,ind)
        n = len(ind)

        a1= xint[j]
        fa1 = f(a1)
        b1 = xint[j+1]
        fb1 = f(b1)

        for kk in range(n):
            # boundary condition for clamped
            if j == 0 and kk == 0 and clamp:
                yeval[0] = fa1
            if j == Nint - 1 and kk == n-1 and clamp:
                yeval[ind[kk]] = fb1
            #line = getLine([a1, fa1], [b1, fb1])
            else:
                yeval[ind[kk]] = getCubic(a1, fa1, b1, fb1, M[j], M[j+1], vals[kk])

    return yeval

f1 = lambda x: 1. / (1 + x**2)
f1p = lambda x: -2 * x / ((1 + x**2)**2)

def f2(x): 
    return x**2
def f2p(x):
    return 2*x

a = -5
b = 5

ns = [5, 10, 15, 20]
N = ns[3]

Neval = 1000

xeval = np.linspace(a,b,Neval)
yevalL = np.zeros(Neval)
yevalH = np.zeros(Neval)

''' create vector with exact values'''

fex = np.zeros(Neval)
for kk in range(Neval):
    fex[kk] = f1(xeval[kk])


''' create equispaced interpolation nodes'''
#xint = np.linspace(a,b,N+1)
# or Chebychev nodes
xint = [np.cos((2*j - 1)*np.pi/(2*(N+1))) for j in range(1,N+2)]
xint = np.array([np.interp(x, [-1,1],[a,b]) for x in xint])
#print(len(xint),'\n',xint)

''' create interpolation data'''
yint = np.zeros(N)
for jj in range(N):
    yint[jj] = f1(xint[jj])

# and derivative interpolation data
yintp = np.zeros(N)
for i in range(N):
    yintp[i] = f1p(xint[i])

y2 = f2(xint)
y2p = f2p(xint)

# get data from interpolation
for i in range(Neval):
    yevalL[i] = eval_lagrange(xeval[i], xint, yint, N)
    yevalH[i] = hermite(xeval[i], xint, yint, yintp, N)
yevalN = eval_cub_spline(xeval, Neval, a, b, f1, N, clamp=False)
yevalC = eval_cub_spline(xeval, Neval, a, b, f1, N)


# set up plotting
fig, ax = plt.subplots(2, 4)
fig.suptitle("With Chebychev Nodes...")
ax[0][0].plot(xeval, yevalL, '-r')
ax[0][0].plot(xeval, fex, color="gray", linestyle="dashed")
ax[0][0].set_title("Lagrange Approximation with N = " + str(N))
ax[1][0].plot(xeval, abs(yevalL - fex))
ax[1][0].set_title("Error")

ax[0][1].plot(xeval, yevalH, '-r')
ax[0][1].plot(xeval, fex, color="gray", linestyle="dashed")
ax[0][1].set_title("Hermite Approximation with N = " + str(N))
ax[1][1].plot(xeval, abs(yevalH - fex))
ax[1][1].set_title("Error")

ax[0][2].plot(xeval, yevalN, '-r')
ax[0][2].plot(xeval, fex, color="gray", linestyle="dashed")
ax[0][2].set_title("Natural Spline Approximation with N = " + str(N))
ax[1][2].plot(xeval, abs(yevalN - fex))
ax[1][2].set_title("Error")

ax[0][3].plot(xeval, yevalC, '-r')
ax[0][3].plot(xeval, fex, color="gray", linestyle="dashed")
ax[0][3].set_title("Clamped Spline Approximation with N = " + str(N))
ax[1][3].plot(xeval, abs(yevalC - fex))
ax[1][3].set_title("Error")

plt.savefig("hw8_n20c.pdf")
plt.show()