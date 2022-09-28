import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mypkg.Iteration1D import Iteration1D

def f(x):
    return np.cos(x)

h = 0.01 * 2.**(-np.arange(0,10))
print("h:", h)

def fp_fw(s, h):
    return (f(s + h) - f(s)) / h

def fp_ct(s, h):
    return (f(s + h) - f(s - h)) / (2*h)

# f'(pi/2) = -sin(pi/2) = -1
x = np.pi / 3
appxs_fw = fp_fw(x, h)
appxs_ct = fp_ct(x, h)

print("The forward approximations are:", appxs_fw)
print("The centered approximations are:", appxs_ct)
