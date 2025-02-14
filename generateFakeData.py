import scipy as scp
import numpy as np 
import math

def sQPD_coherent(s, alpha, alpha0=0):
    return 2/(np.pi * (1 - s)) * np.exp(-2*np.abs(alpha - alpha0)**2/(1-s))

def sQPD_thermal(s, alpha, N0):
    return 2/(np.pi * (2 * (N0 + 1/2) - s) ) * np.exp(-2*np.abs(alpha)**2/(2 * (N0 + 1/2) -s ))

def sQPD_fock(s, alpha, n):
    if s==-1:
        return 1/np.pi * np.abs(alpha)**(2*n) / math.factorial(n) * np.exp(-np.abs(alpha)**2)
    else:
        return (2/(np.pi * (1 - s) ) * ((s+1)/(s-1))**n * scp.special.eval_laguerre(n, 4*np.abs(alpha)**2/(1 - s**2)) * np.exp(-2*np.abs(alpha)**2/(1 -s )))
    
def hit_or_miss(N, X, Y, P):
    Xmin, Xmax, Ymin, Ymax = np.min(X), np.max(X), np.min(Y), np.max(Y)
    samples = []
    k = 0
    while k < N:
        x = np.random.uniform(Xmin, Xmax, 1)
        y = np.random.uniform(Ymin, Ymax, 1)
        z = np.random.uniform(np.min(P), np.max(P), 1)
        i = np.abs(x - X).argmin()
        j = np.abs(y - Y).argmin()
        if z > P[i, j]:
            continue
        k += 1
        samples.append(x + 1j*y)
        
    return np.array(samples).squeeze()