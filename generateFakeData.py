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


def f_coherent(x, y, s, alpha0):
    alpha = x + 1j*y
    return 2/(np.pi * (1 - s)) * np.exp(-2*np.abs(alpha - alpha0)**2/(1-s))

def f_thermal(x, y, s, N0):
    alpha = x + 1j*y
    return 2/(np.pi * (2 * (N0 + 1/2) - s) ) * np.exp(-2*np.abs(alpha)**2/(2 * (N0 + 1/2) -s ))

def f_fock(x, y, s, n):
    alpha = x + 1j*y
    if s==-1:
        return 1/np.pi * np.abs(alpha)**(2*n) / math.factorial(n) * np.exp(-np.abs(alpha)**2)
    else:
        return (2/(np.pi * (1 - s) ) * ((s+1)/(s-1))**n * scp.special.eval_laguerre(n, 4*np.abs(alpha)**2/(1 - s**2)) * np.exp(-2*np.abs(alpha)**2/(1 -s )))
    

def sample_from_analytic_distribution(N, p_func, domain, M, batch_size=100000, **p_kwargs):
    """
    Sample exactly N points from a 2D distribution defined by an analytic function.
    
    Parameters:
        N (int): Number of accepted samples desired.
        p_func (callable): A function of two variables, p(x, y, **p_kwargs), that returns the density.
                           It should be vectorized (i.e., accept NumPy arrays as input).
        domain (tuple): The sampling domain defined as (x_min, x_max, y_min, y_max).
        M (float): An upper bound for p(x, y) over the domain.
        batch_size (int): Number of candidate points generated per batch.
        **p_kwargs: Additional keyword arguments to be passed to p_func.
    
    Returns:
        accepted_xs (1D np.array): x-coordinates of the accepted samples.
        accepted_ys (1D np.array): y-coordinates of the accepted samples.
    """
    x_min, x_max, y_min, y_max = domain
    accepted_xs_list = []
    accepted_ys_list = []
    total_accepted = 0

    while total_accepted < N:
        # Generate candidate points uniformly over the domain.
        xs = np.random.uniform(x_min, x_max, batch_size)
        ys = np.random.uniform(y_min, y_max, batch_size)
        
        # Evaluate the density function at these points with extra parameters.
        p_vals = p_func(xs, ys, **p_kwargs)
        
        # Generate uniform random numbers for the acceptance test.
        us = np.random.uniform(0, M, batch_size)
        
        # Accept the candidate points where u < p(x, y)
        mask = us < p_vals
        accepted_xs_list.append(xs[mask])
        accepted_ys_list.append(ys[mask])
        total_accepted += np.sum(mask)

    # Concatenate accepted samples from all batches.
    accepted_xs = np.concatenate(accepted_xs_list)
    accepted_ys = np.concatenate(accepted_ys_list)
    
    # Trim to exactly N accepted samples.
    return accepted_xs[:N] + 1j* accepted_ys[:N]