import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sci
from timeit import default_timer as timer

# Look into applications
# Compare Ill-conditioned Jacobians
# compare to numpy rootfinder

# Rough draft needs outline for what still needs to be completed
# At least one table and one plot of results

def compute_order(x,xstar):
    # p_{n+1}-p (from the second index to the end)
    diff1 = np.abs(x[1::]-xstar)
    # p_n-p (from the first index to the second to last)
    diff2 = np.abs(x[0:-1]-xstar)
    # linear fit to log of differences
    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    print('the order equation is')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
    print('lambda = ' + str(np.exp(fit[1])))
    print('alpha = ' + str(fit[0]))
    return [fit,diff1,diff2]

# Non-approximating root finders
def grad_descent(f, x0, G, tol, nmax):
    xn = x0
    xstar = {}

    xnm1 = None
    gradm1 = None
    for i in range(nmax):
        grad = np.array([f(xn) for f in G])

        # find another way to calculate step size (from code)
        if i == 0:
            stepSize = 0.01
        else:
            gradDiff = np.array([grad-gradm1])
            stepSize = np.linalg.norm(np.matmul(np.array([xn - xnm1]).T, gradDiff)) / np.linalg.norm(gradDiff)**2

        xn1 = xn - stepSize*grad

        fx = f(xn1)
        xstar[fx] = xn1

        gp1 = np.array([f(xn1) for f in G])
        if(np.linalg.norm(gp1) < tol):
            return fx, xn1, i+1, xstar
        
        xnm1 = xn
        gradm1 = grad
        xn = xn1
    return f(xn1), xn1, i+1, xstar


def nd_descent(f, x0, G, H, tol, nmax):
    xn = x0
    xstar = {}
    for i in range(nmax):
        grad = np.array([f(xn) for f in G])
        hess = [[f(xn) for f in H[r]] for r in range(len(H))]
        
        hinv = np.linalg.inv(hess)

        xn1 = xn - np.matmul(hinv, grad)

        fx = f(xn1)
        xstar[fx] = xn1

        gp1 = np.array([f(xn1) for f in G])

        if(np.linalg.norm(gp1) < tol):
            return fx, xn1, i+1, xstar
        
        xn = xn1
    return f(xn1), xn1, i+1, xstar


# Approximating root finders
def lazy_nd(f, x0, G, H, tol, nmax):
    xn = x0
    xstar = []
    for i in range(nmax):
        grad = np.array([f(xn) for f in G])
        
        # calculate hessian every 4 iterations instead of every iteration
        if i%4 == 0:
            hess = [[f(xn) for f in H[r]] for r in range(len(H))]
            hinv = np.linalg.inv(hess)

        xn1 = xn - np.matmul(hinv, grad)

        fx = f(xn1)
        xstar.append(fx)

        gp1 = np.array([f(xn1) for f in G])

        if(np.linalg.norm(gp1) < tol):
            return fx, xn1, i+1, xstar[:-1]
        
        xn = xn1
    return f(xn1), xn1, i+1, xstar[:-1]

def BFGS_nd(f, x0, G, H, tol, nmax):
    xn = x0
    gn = np.array([f(xn) for f in G])
    xstar = []
    
    def bt_linesearch(x, p, f, grad, t, alpha=0.3, beta=0.9):
        gradTp = np.dot(grad, p)
        while f(x + (t * p)) > f(x) + (alpha*t * gradTp):
            t *= beta
        return t

    for i in range(nmax):
        # do inversion on first iteration
        if i == 0:
            #hess = [[f(xn) for f in H[r]] for r in range(len(H))]
            hinv = np.identity(len(H))
            xn1 = xn - np.matmul(hinv, gn)
            gn1 = np.array([f(xn1) for f in G])
        # rank 1 appx on all other iterations
        else:
            p_n = np.matmul(hinv, (gn*-1)) # solve for direction
            # print(p_n)
            # print(gn)
            alpha_n = bt_linesearch(xn, p_n, f, gn, 1)  # do inexact line search to get stepsize
            snT = np.array([alpha_n * p_n])
            sn = snT.T
            xn = np.array([xn]).T
            xn1 = sn + xn
            gn1 = np.array([f(xn1) for f in G])
            yn = np.array([gn1.T[0] - gn]).T
            # print("G_n+1:")
            # print(gn1)
            # print("y_n:")
            # print(yn)
            ynT = yn.T

            # if np.linalg.norm(yn)/np.linalg.norm(gn1) < tol:
            #     fx = f(xn1.T[0])
            #     xstar.append(fx)
            #     return fx, xn1.T[0], i+1, xstar[:-1]

            # apply Sherman-Morrison to approximate hessian inverse
            snTyn = np.matmul(snT, yn)
            ynTB = np.matmul(ynT, hinv)
            sum1 = snTyn + np.matmul(ynTB, yn)
            #print(sum1)
            snsnT = np.matmul(sn, snT)
            #print(snsnT)
            snTyn2 = snTyn**2
            d1 = sum1 * snsnT / snTyn2
            d2 = (np.matmul(np.matmul(hinv, yn), snT) + np.matmul(np.matmul(sn, ynT), hinv)) / snTyn
            hinv = hinv + d1 - d2
            
        if i >= 1:
            xn1 = xn1.T[0]
            
        fx = f(xn1)
        xstar.append(fx)

        if(np.linalg.norm(gn1) < tol):
            return fx, xn1, i+1, xstar[:-1]
        
        xn = xn1
        if i == 0:
            gn = gn1
        else:
            gn = gn1.T[0]
    return f(xn1), xn1, i+1, xstar[:-1]


def driver():
    # This may or may not be "well enough represented by a quadratic model"
    f1 = lambda x: (x[0]-3)**2 + (x[1]+2)**2 + x[2]**2

    g1 = [lambda x: 2*(x[0]-3),
          lambda x: 2*(x[1]+2),
          lambda x: 2*x[2]]

    h1 = [
        [lambda x: 2, lambda x: 0, lambda x: 0],
        [lambda x: 0, lambda x: 2, lambda x: 0],
        [lambda x: 0, lambda x: 0, lambda x: 2]
    ]

    f2 = lambda x: 3*x[0]**2 - np.cos(x[1]*x[2]) - 1.5

    g2 = [
        lambda x: 6*x[0],
        lambda x: -x[2]*np.sin(x[1]*x[2]),
        lambda x: -x[1]*np.sin(x[1]*x[2])
    ]

    h2 = [
        [lambda x: 6, lambda x: 0, lambda x: 0],
        [lambda x: 0, lambda x: -x[2]**2 * np.cos(x[1]*x[2]), lambda x: -np.sin(x[1]*x[2]) - x[1]*x[2]*np.cos(x[1]*x[2])],
        [lambda x: 0, lambda x: -np.sin(x[1]*x[2]) - x[1]*x[2]*np.cos(x[1]*x[2]), lambda x: -x[1]**2 * np.cos(x[1]*x[2])]
    ]

    f3 = lambda x: (8*x[0] + x[1] + 7*x[2] + 2*x[3] + 7*x[4] + x[5] - 1)**2 + (7*x[0] + 3*x[1] + 5*x[2] + 1*x[3] + 8*x[4] + 3*x[5] - 2)**2 + (3*x[0] + 7*x[1] + 7*x[2] + 9*x[3] + 2*x[4] + x[5] - 3)**2 + (5*x[0] + 7*x[1] + 6*x[2] + 9*x[3] + 3*x[4] + 3*x[5] - 4)**2 + (2*x[0] + 3*x[1] + 6*x[2] + 8*x[3] + 3*x[4] + 9*x[5] - 5)**2 + (3*x[0] + 9*x[1] + 4*x[2] + 7*x[3] + 3*x[4] - 6)**2
    
    g3 = [
        lambda x: 16 * (8*x[0] + x[1] + 7*x[2] + 2*x[3] + 7*x[4] + x[5] - 1) + 14 * (7*x[0] + 3*x[1] + 5*x[2] + 1*x[3] + 8*x[4] + 3*x[5] - 2) + 6  * (3*x[0] + 7*x[1] + 7*x[2] + 9*x[3] + 2*x[4] + x[5] - 3) + 10 * (5*x[0] + 7*x[1] + 6*x[2] + 9*x[3] + 3*x[4] + 3*x[5] - 4) + 4  * (2*x[0] + 3*x[1] + 6*x[2] + 8*x[3] + 3*x[4] + 9*x[5] - 5) + 6  * (3*x[0] + 9*x[1] + 4*x[2] + 7*x[3] + 3*x[4] - 6),
        lambda x: 2  * (8*x[0] + x[1] + 7*x[2] + 2*x[3] + 7*x[4] + x[5] - 1) + 6  * (7*x[0] + 3*x[1] + 5*x[2] + 1*x[3] + 8*x[4] + 3*x[5] - 2) + 14 * (3*x[0] + 7*x[1] + 7*x[2] + 9*x[3] + 2*x[4] + x[5] - 3) + 14 * (5*x[0] + 7*x[1] + 6*x[2] + 9*x[3] + 3*x[4] + 3*x[5] - 4) + 6  * (2*x[0] + 3*x[1] + 6*x[2] + 8*x[3] + 3*x[4] + 9*x[5] - 5) + 18 * (3*x[0] + 9*x[1] + 4*x[2] + 7*x[3] + 3*x[4] - 6),
        lambda x: 14 * (8*x[0] + x[1] + 7*x[2] + 2*x[3] + 7*x[4] + x[5] - 1) + 10 * (7*x[0] + 3*x[1] + 5*x[2] + 1*x[3] + 8*x[4] + 3*x[5] - 2) + 14 * (3*x[0] + 7*x[1] + 7*x[2] + 9*x[3] + 2*x[4] + x[5] - 3) + 12 * (5*x[0] + 7*x[1] + 6*x[2] + 9*x[3] + 3*x[4] + 3*x[5] - 4) + 12 * (2*x[0] + 3*x[1] + 6*x[2] + 8*x[3] + 3*x[4] + 9*x[5] - 5) + 8  * (3*x[0] + 9*x[1] + 4*x[2] + 7*x[3] + 3*x[4] - 6),
        lambda x: 4  * (8*x[0] + x[1] + 7*x[2] + 2*x[3] + 7*x[4] + x[5] - 1) + 2  * (7*x[0] + 3*x[1] + 5*x[2] + 1*x[3] + 8*x[4] + 3*x[5] - 2) + 18 * (3*x[0] + 7*x[1] + 7*x[2] + 9*x[3] + 2*x[4] + x[5] - 3) + 18 * (5*x[0] + 7*x[1] + 6*x[2] + 9*x[3] + 3*x[4] + 3*x[5] - 4) + 16 * (2*x[0] + 3*x[1] + 6*x[2] + 8*x[3] + 3*x[4] + 9*x[5] - 5) + 14 * (3*x[0] + 9*x[1] + 4*x[2] + 7*x[3] + 3*x[4] - 6),
        lambda x: 14 * (8*x[0] + x[1] + 7*x[2] + 2*x[3] + 7*x[4] + x[5] - 1) + 16 * (7*x[0] + 3*x[1] + 5*x[2] + 1*x[3] + 8*x[4] + 3*x[5] - 2) + 4  * (3*x[0] + 7*x[1] + 7*x[2] + 9*x[3] + 2*x[4] + x[5] - 3) + 6  * (5*x[0] + 7*x[1] + 6*x[2] + 9*x[3] + 3*x[4] + 3*x[5] - 4) + 6  * (2*x[0] + 3*x[1] + 6*x[2] + 8*x[3] + 3*x[4] + 9*x[5] - 5) + 6  * (3*x[0] + 9*x[1] + 4*x[2] + 7*x[3] + 3*x[4] - 6),
        lambda x: 2 *  (8*x[0] + x[1] + 7*x[2] + 2*x[3] + 7*x[4] + x[5] - 1) + 6  * (7*x[0] + 3*x[1] + 5*x[2] + 1*x[3] + 8*x[4] + 3*x[5] - 2) + 2  * (3*x[0] + 7*x[1] + 7*x[2] + 9*x[3] + 2*x[4] + x[5] - 3) + 6  * (5*x[0] + 7*x[1] + 6*x[2] + 9*x[3] + 3*x[4] + 3*x[5] - 4) + 18 * (2*x[0] + 3*x[1] + 6*x[2] + 8*x[3] + 3*x[4] + 9*x[5] - 5)
    ]

    h3 = [
        [lambda x: 320, lambda x: 236, lambda x: 332, lambda x: 264, lambda x: 296, lambda x: 130],
        [lambda x: 236, lambda x: 396, lambda x: 334, lambda x: 436, lambda x: 204, lambda x: 130],
        [lambda x: 332, lambda x: 334, lambda x: 422, lambda x: 424, lambda x: 302, lambda x: 202],
        [lambda x: 264, lambda x: 436, lambda x: 424, lambda x: 560, lambda x: 224, lambda x: 226],
        [lambda x: 296, lambda x: 204, lambda x: 302, lambda x: 224, lambda x: 288, lambda x: 138],
        [lambda x: 130, lambda x: 130, lambda x: 202, lambda x: 226, lambda x: 138, lambda x: 202]

    ]

    x02 = [1, -0.5, 0.75]
    initial_guess = [0,0,0,0,0,0]
    tol = 1e-8; nmax = 1000

    root = np.float64(-2.5)

    t1 = timer()
    gd_val, gd_root, gd_iters, gd_xstar = grad_descent(f2, x02, g2, tol, nmax)
    dt = timer() - t1
    print("Steepest Descent root:", gd_root)
    print("f(", gd_root, ") =", gd_val)
    print("Absolute Error:", abs(gd_val - root))
    print("Iterates:", gd_xstar)
    print("This took",gd_iters,"iterations and",dt,"seconds\n")

    gx = list(gd_xstar.values())
    q = np.log(np.linalg.norm(gx[-1] - gx[-2])/np.linalg.norm(gx[-2] - gx[-3])) / np.log(np.linalg.norm(gx[-2] - gx[-3])/np.linalg.norm(gx[-3] - gx[-4]))
    print("analytic order of convergence:", q)
    # print("Delta x11 =",np.linalg.norm(gd_root - list(gd_xstar.values())[10])**2)
    # print("Delta x12 =",np.linalg.norm(gd_root - list(gd_xstar.values())[11])**2)
    n = 1
    s = 0
    for i in range(n):
        t1 = timer()
        nd_val, nd_root, nd_iters, nd_xstar = nd_descent(f2, x02, g2, h2, tol, nmax)
        dt = timer() - t1
        s += dt
    dt = s / n
    print("Newton Direction root:", nd_root)
    #print(nd_root[1]*nd_root[2])
    print("f(", nd_root, ") =", nd_val)
    print("Absolute Error:", abs(nd_val - root))
    print("Iterates:", nd_xstar)
    # print("Delta x13 =",np.linalg.norm(nd_root - list(nd_xstar.values())[12])**2)
    # print("Delta x14 =",np.linalg.norm(nd_root - list(nd_xstar.values())[13])**2)
    print("This took",nd_iters,"iterations and",dt,"seconds\n")

    nx = list(nd_xstar.values())
    q = np.log(np.linalg.norm(nx[-2] - nx[-3])/np.linalg.norm(nx[-3] - nx[-4])) / np.log(np.linalg.norm(nx[-3] - nx[-4])/np.linalg.norm(nx[-4] - nx[-5]))
    print("analytic order of convergence:", q)

    # t1 = timer()
    # lazy_val, lazy_root, lazy_iters, lazy_xstar = lazy_nd(f3, initial_guess, g3, h3, tol, nmax)
    # dt = timer() - t1
    # print("Lazy Newton Direction root:", lazy_root)
    # #print(lazy_root[1]*lazy_root[2])
    # print("f(", lazy_root, ") =", lazy_val)
    # print("Absolute Error:", abs(lazy_val - root))
    # #print("Iterates:", lazy_xstar)
    # print("This took",lazy_iters,"iterations and",dt,"seconds\n")
    # s = 0
    # for i in range(n):
    #     t1 = timer()
    #     bfgs_val, bfgs_root, bfgs_iters, bfgs_xstar = BFGS_nd(f3, initial_guess, g3, h3, tol, nmax)
    #     dt = timer() - t1
    #     s += dt
    # dt = s / n
    # print("BFGS Newton Direction root:", bfgs_root)
    # #print(bfgs_root[1]*bfgs_root[2])
    # print("f(", bfgs_root, ") =", bfgs_val)
    # print("Absolute Error:", abs(bfgs_val - root))
    # #print("Iterates:", bfgs_xstar)
    # print("This took",bfgs_iters,"iterations and",dt,"seconds\n")

    
    # print('\nrate and constants for Steepest Descent')
    # [fit,diff1,diff2] = compute_order(list(gd_xstar.keys())[10:12], root)
    # print('\nrate and constants for Newton Direction')
    # [fit1,diff11,diff21] = compute_order(list(nd_xstar.keys())[:-1], root)
    # # print('\nrate and constants for BFGS Newton Direction')
    # # [fit2,diff12,diff22] = compute_order(bfgs_xstar, root)
    # # plot the data
    # plt.loglog(diff2,diff1,'gx',label='Steepest Descent data')
    # plt.loglog(diff21,diff11,'rx',label='Newton Direction data')
    # # plt.loglog(diff22,diff12,'x', color="orange",label='BFGS Newton Direction data')
    # # plot the fits
    # plt.loglog(diff2,np.exp(fit[1]+fit[0]*np.log(diff2)),'g-',label='SD fit')
    # plt.loglog(diff21,np.exp(fit1[1]+fit1[0]*np.log(diff21)),'b-',label='ND fit')
    # # plt.loglog(diff22,np.exp(fit2[1]+fit2[0]*np.log(diff22)),'-',color="orange",label='BFGS fit')
    # plt.title("Empirical Convergence of SD for Function 2")
    # # label the plot axes and create legend
    # plt.xlabel('$|p_{n}-p|$')
    # plt.ylabel('$|p_{n+1}-p|$')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    driver()
