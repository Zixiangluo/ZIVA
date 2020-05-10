import numpy as np
from scipy.optimize import curve_fit

def func(s,km):
    p = 1 - s / (km + s)
    return p

def func2(u,k):
    p = np.exp(-k*np.square(u))
    return p

def fitvars(count):
    mean_expr = np.mean(count, axis=0)
    dropout_rate = np.sum(count == 0, axis=0) / count.shape[0]
    return mean_expr, dropout_rate

def fit(count, f = "nb"):
    mean_expr, dropout_rate = fitvars(count)
    if(f=="nb"):
        lmd, _ = curve_fit(func2, mean_expr, dropout_rate)
    elif(f=="mm"):
        lmd, _ = curve_fit(func, mean_expr, dropout_rate)
    else:
        return False
    return lmd