from mypkg.Iteration1D import Iteration1D

# Problem 1
f = lambda x: x**2 * (x-1)

find = Iteration1D(f,'bisection')
find.tol = 1e-6; find.Nmax = 100

# part A
find.a = 0.5; find.b = 2
x = find.root()
print("For the interval:", find.a, "to", find.b, ", the root is x =", x, find.info)


# part B
find.a = -1; find.b = 0.5
x = find.root()
# This interval does not have a successful answer since the root is not in the given interval
# 
print("For the interval:", find.a, "to", find.b, ", the root is x =", x, find.info)


# part C
find.a = -1; find.b = 2
x = find.root()
print("For the interval:", find.a, "to", find.b, ", the root is x =", x, find.info)

# try getting the root x = 0
find.a = 0; find.b = 0.5
x = find.root()
print("For the interval:", find.a, "to", find.b, ", the root is x =", x, find.info)
# when the lower bound of the interval is 0 and the interval doesn't contain the root
# the algorithm just returns the lower bound so we get x = 0

find.a = -0.5; find.b = 0.5
x = find.root()
print("For the interval:", find.a, "to", find.b, ", the root is x =", x, find.info)
# however, the algorithm doesn't normally find the root x = 0 and returns an error


# Problem 2
find.method = 'fixedpt'
find.p0 = 1

# recast the problem
# idk how to do this yet
x = find.root()

print("The root is x =", x, find.info)
