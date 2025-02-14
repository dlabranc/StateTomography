import numpy as np
from math import factorial, comb, sqrt
import sympy as sp
from joblib import Parallel, delayed

def compute_contribution(measurement, n, m):
    return (np.conjugate(measurement) ** n) * (measurement ** m)

def calculate_moments_parallel(S_measurements, max_order, n_jobs=-1):
    """
    Calculate moments in parallel from S_measurements using joblib.
    
    Parameters:
      S_measurements (np.ndarray): Array of complex S measurements.
      max_order (int): Maximum order of moments to calculate.
      n_jobs (int): Number of parallel jobs (default: -1 uses all available cores).
      
    Returns:
      np.ndarray: Array of moments with shape (max_order+1, max_order+1).
    """
    num_samples = len(S_measurements)
    moments = np.zeros((max_order + 1, max_order + 1), dtype=complex)
    
    for n in range(max_order + 1):
        for m in range(max_order + 1):
            # Parallelize the sum over measurements for each (n, m)
            contributions = Parallel(n_jobs=n_jobs)(
                delayed(compute_contribution)(measurement, n, m)
                for measurement in S_measurements
            )
            moment_sum = np.sum(contributions)
            moments[n, m] = moment_sum / num_samples
            print(f'\rCalculated moment ({n},{m})', end='')
    print()  # Newline after progress output
    return moments

def calculate_moments(S_measurements, max_order):
    """
    Calculate the moments of S from the given measurements.

    Parameters:
    S_measurements (np.ndarray): Array of S measurements (shape: num_samples).
    max_order (int): The maximum order of moments to calculate.

    Returns:
    np.ndarray: Array of moments up to the specified order.
    """
    num_samples = len(S_measurements)
    moments = np.zeros((max_order + 1, max_order + 1))
    index = 0
    for n in range(max_order + 1):
        for m in range(max_order + 1):
            index = 1 + index
            print(f'\rCalculating moment of order {n} {m} (Max order {max_order} {max_order})\t Percentage of completion: {100*(index)/(max_order+1)**2:.2f}%', end='\r')

            moment_sum = 0
            
            for measurement in S_measurements:
                moment_sum += (np.conj(measurement) ** n) * (measurement ** m)
            moments[n, m] = 1/(num_samples) * np.abs(moment_sum)
            '''uncertainty_sum = 0
            for measurement in S_measurements:
                uncertainty_sum += ((np.conj(measurement) ** n) * (measurement ** m) - moments[n, m])**2
            uncertainties[n, m] = np.sqrt(1/(num_samples) * np.abs(uncertainty_sum))
'''
    print()
    return moments


def extract_a_moments(S_moments, h_moments, max_order):
    # Define symbols for the a_moments
    a_moments = sp.IndexedBase('a_moments')
    
    # Initialize the equation
    equations = []
    for n in range(max_order + 1):
        for m in range(max_order + 1):
            equation = 0
            for i in range(n + 1):
                for j in range(m + 1):
                    equation += sp.binomial(m, j) * sp.binomial(n, i) * a_moments[i, j] * h_moments[n-i, m-j]
            eq = sp.Eq(S_moments[n,m], equation)

            equations.append(eq)
    
    # Equation given in the problem
    
    # Solve the equation for a_moments
    a_symbols = [a_moments[i, j] for i in range(n + 1) for j in range(m + 1)]
    solution = sp.solve(equations, a_symbols)
    a_pi = np.zeros(shape=(max_order+1, max_order+1))

    for i in range(max_order+1):
        for j in range(max_order+1):
            a_pi[i][j] = solution[sp.IndexedBase('a_moments')[i, j]]
    return a_pi


def build_density_matrix(signal_moments, N, L_max):
    """
    Build the density matrix in the Fock basis using the formula:
    
      <n|ρ|m> = 1/sqrt(n! m!) * Σ_{l=0}^{L_max} [ (-1)^l / l! * <(a†)^(n+l) a^(m+l)> ]
    
    Parameters:
      signal_moments : dict
          Dictionary with keys (n, m) corresponding to <(a†)^n a^m>.
      N : int
          Dimension of the truncated Hilbert space (states |0>, ..., |N-1>).
      L_max : int
          Maximum value of l in the summation.
    
    Returns:
      rho : ndarray
          The reconstructed density matrix as an N x N numpy array.
    """
    rho = np.zeros((N, N), dtype=complex)
    for n in range(N):
        for m in range(N):
            sum_val = 0.0
            for l in range(L_max + 1):
                if n+l < signal_moments.shape[0] and m+l < signal_moments.shape[1]:
                    moment = signal_moments[n+l, m+l]
                else:
                    moment = 0
                sum_val += ((-1) ** l / factorial(l)) * moment
            rho[n, m] = sum_val / (sqrt(factorial(n) * factorial(m)))
    # Normalize the density matrix to have unit trace
    trace = np.trace(rho)
    if abs(trace) > 1e-12:
        rho = rho / trace
    return rho


def g2_from_density_matrix(rho):
    """
    Calculate the second-order correlation g^(2)(0) from a density matrix in the Fock basis.
    
    g^(2)(0) = <n(n-1)> / (<n>)^2
              = [sum_{n} n*(n-1)*rho[n,n]] / ( [sum_{n} n*rho[n,n]]^2 )
    
    Parameters:
      rho : numpy.ndarray
          Density matrix in the Fock basis (assumed truncated to finite dimension).
    
    Returns:
      g2 : float
          The computed second-order correlation.
    """
    N = rho.shape[0]
    n_avg = 0.0
    n_n_minus1 = 0.0
    for n in range(N):
        # Use the real part of the diagonal elements (they should be real)
        prob = np.real(rho[n, n])
        n_avg += n * prob
        n_n_minus1 += n * (n - 1) * prob
    if n_avg == 0:
        return 0.0
    return n_n_minus1 / (n_avg ** 2)

def g2_from_moments(moments):
    """
    Calculate the second-order correlation g^(2)(0) from measured normally-ordered moments.
    
    Here, moments is a dictionary with keys (n, m) corresponding to <(a†)^m a^n>.
    For g^(2)(0), we use:
       g^(2)(0) = <(a†)^2 a^2> / (<a† a>)^2
    
    Parameters:
      moments : dict
          Dictionary with keys (n, m) for moments.
          It must include (1,1) and (2,2) entries.
    
    Returns:
      g2 : float
          The computed second-order correlation.
    """
    n_avg = moments[1,1]
    moment_22 = moments[2,2]
    if n_avg == 0:
        return 0.0
    return moment_22 / (n_avg ** 2)