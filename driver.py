import num_algos as na
import numpy as np
import matplotlib as plt

def f(x):
    return np.sin(x)

print("Root of f:")
print(na.bisection(f, 3, 4, 10**(-17)))
