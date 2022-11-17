import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


def comp_trap(a, b, n, f, xnodes):
    h = (b-a)/n
    middle_sum = 2*sum(f(xnodes[j]) for j in range(1, n))
    return h/2 * (0 + middle_sum + f(b))

def comp_simpsons(a, b, n, f, xnodes):
    h = (b-a)/n
    even_sum = 2*sum(f(xnodes[2*j]) for j in range(1,int(n/2) + 1))
    odd_sum = 4*sum(f(xnodes[2*j-1]) for j in range(1,int(n/2) + 1))
    return h/3 * (0 + even_sum + odd_sum + f(b))

def driver():
    f = lambda s: np.cos(1/s)*s**3

    a = 0; b = 1

    tol = 1e-4
    print("Tol = " + str(tol))
    
    #xeval = np.linspace(a, b, 1000)

    # Trapezoidal
    n = 5
    xnodes = np.linspace(a, b, n+1)
    trap_appx = comp_trap(a, b, n, f, xnodes)
    print("The composite trapezoidal approximation is:", trap_appx)
    print("This took",n,"evaluations\n")

    # Simpsons
    n = 5
    xnodes = np.linspace(a, b, n+1)
    simp_appx = comp_simpsons(a, b, n, f, xnodes)
    print("The composite Simpson's approximation is:", simp_appx)
    print("This took",n,"evaluations\n")

    # Quad
    quad_out = spi.quad(f, a, b, epsabs=tol, full_output=True)
    infodict = quad_out[2]
    neval = infodict["neval"]

    quad = quad_out[0]

    print("SCIPY quad integral is:", quad)
    print("This took SCIPY",neval,"function evals\n")
    
    trap_diff = abs(trap_appx - quad)
    print("The error for trap is:", trap_diff)

    simp_diff = abs(simp_appx - quad)
    print("The error for Simpsons is:", simp_diff)

    return


if __name__ == '__main__':
    driver()