import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
import scipy.integrate as spi

a = 0; b = 200
n = 5000
tol = 1e-6
xvals = [2, 4, 6, 8, 10]

for myX in xvals:
    print("For x =",myX,":")
    def f(x, t):
        return t**(x-1)*np.exp(-t)

    def f_helper(t):
        return f(myX, t)

    def f_lg(t):
        return t**(myX - 1)

    def comp_trap(a, b, n, f):
        h = (b-a)/n
        xnodes = np.linspace(a, b, n+1)
        middle_sum = 2*sum(f(xnodes[j]) for j in range(1, n))
        return h/2 * (f(a) + middle_sum + f(b))


    trap_appx = comp_trap(a, b, n, f_helper)
    print("The composite trapezoidal approximation is:", trap_appx)
    print("This took",n,"evaluations\n")

    # SCIPY special
    gamma = sps.gamma(myX)

    print("SCIPY gamma output is:", gamma)
    print()

    # Quad
    quad_out = spi.quad(f_helper, a, b, epsabs=tol, full_output=True)
    infodict = quad_out[2]
    neval = infodict["neval"]

    quad = quad_out[0]

    print("SCIPY quad integral is:", quad)
    print("This took SCIPY",neval,"function evals\n")

    trap_diff = abs(trap_appx - gamma)
    print("The error for trap is:", trap_diff)

    quad_diff = abs(trap_appx - quad)
    print("The error for trap from quad is:", quad_diff)
    
    samples, weights = np.polynomial.laguerre.laggauss(100)
    gl_appx = sum(weights[i]*f_lg(samples[i]) for i in range(len(weights)))
    print("\nThe Gauss-Laguerre approximation is:",gl_appx)
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


