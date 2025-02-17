import numpy as np
from math import factorial, comb, sqrt
import sympy as sp
import cvxpy as cvx
from joblib import Parallel, delayed
#  
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

def calculate_moments_uncertainty(S_measurements, max_order):
    """
    Calculate the moments of S from the given measurements and their standard deviations.

    Parameters:
      S_measurements (np.ndarray): Array of S measurements (shape: num_samples).
      max_order (int): The maximum order of moments to calculate.

    Returns:
      moments (np.ndarray): Array of moments of shape (max_order+1, max_order+1).
      uncertainties (np.ndarray): Array of standard deviations for each moment,
                                  with the same shape.
    """
    num_samples = len(S_measurements)
    moments = np.zeros((max_order + 1, max_order + 1), dtype=complex)
    uncertainties = np.zeros((max_order + 1, max_order + 1), dtype=float)
    total_keys = (max_order + 1) ** 2
    index = 0

    for n in range(max_order + 1):
        for m in range(max_order + 1):
            index += 1
            print(f'\rCalculating moment ({n},{m}) - Completion: {100 * index / total_keys:.2f}%', end='\r')

            moment_sum = 0.0 + 0.0j
            # Sum the (conjugate(measurement)^n * measurement^m) over all measurements
            for measurement in S_measurements:
                moment_sum += (np.conjugate(measurement) ** n) * (measurement ** m)
            mean_value = moment_sum / num_samples

            # Store the calculated moment
            moments[n, m] = np.abs(mean_value)

            # Compute the uncertainty (standard deviation) for this moment
            uncertainty_sum = 0.0
            for measurement in S_measurements:
                diff = (np.conjugate(measurement) ** n) * (measurement ** m) - mean_value
                uncertainty_sum += np.abs(diff)**2
            uncertainties[n, m] = np.sqrt(uncertainty_sum / num_samples)
    print()
    return moments, uncertainties

def extract_a_moments_uncertainty(S_moments, S_uncertainties, h_moments, max_order, num_samples=100):
    """
    Propagate uncertainties from S_moments to the extracted signal moments a_moments
    using a Monte Carlo sampling approach.
    
    For each Monte Carlo iteration, a new S_moments_sample is generated
    by sampling from a Gaussian distribution with mean=S_moments and standard deviation=S_uncertainties.
    The extraction function is then applied to obtain a sample of a_moments.
    
    Parameters:
      S_moments (np.ndarray): Array of measured moments (shape (max_order+1, max_order+1)).
      S_uncertainties (np.ndarray): Array of standard deviations for S_moments (same shape).
      h_moments (dict): Dictionary with keys (n, m) for noise moments.
      max_order (int): Maximum order considered.
      num_samples (int): Number of Monte Carlo samples.
      
    Returns:
      a_mean (np.ndarray): Mean extracted signal moments (shape (max_order+1, max_order+1)).
      a_std (np.ndarray): Standard deviation (uncertainty) of extracted signal moments (same shape).
    """
    # Container for Monte Carlo samples of a_moments
    samples = np.zeros((num_samples, max_order + 1, max_order + 1), dtype=complex)
    
    for k in range(num_samples):
        print(f'\rSample ({k}/{num_samples}) - Completion: {100 * k / num_samples:.2f}%', end='\r')
        # Create a new S_moments_sample by sampling each element
        S_sample = np.zeros_like(S_moments, dtype=complex)
        for i in range(max_order + 1):
            for j in range(max_order + 1):
                # Here we assume S_moments are real (or their magnitude was taken)
                S_sample[i, j] = np.random.normal(S_moments[i, j], S_uncertainties[i, j]).real
        # Extract a_moments from the sampled S_moments
        try:
            a_sample = extract_a_moments(S_sample, h_moments, max_order)
        except Exception as e:
            print(f"Sample {k} extraction failed: {e}")
            a_sample = np.zeros((max_order + 1, max_order + 1), dtype=complex)
        samples[k, :, :] = a_sample

    # Compute mean and standard deviation over the samples
    a_mean = np.mean(samples, axis=0)
    a_std = np.std(samples, axis=0)
    return a_mean, a_std

def sample_a_moments(S_moments, S_uncertainties, h_moments, max_order):
    """
    Generate one Monte Carlo sample of the extracted signal moments.
    For each (i,j), sample S_sample[i,j] from a Gaussian with mean S_moments[i,j]
    and standard deviation S_uncertainties[i,j], then extract a_moments.
    """
    S_sample = np.zeros_like(S_moments, dtype=complex)
    n_rows, n_cols = S_moments.shape
    for i in range(n_rows):
        for j in range(n_cols):
            # Sampling the real part (assuming the moments are real-valued)
            S_sample[i, j] = np.random.normal(S_moments[i, j], S_uncertainties[i, j])
    try:
        a_sample = extract_a_moments(S_sample, h_moments, max_order)
    except Exception as e:
        print(f"Sample extraction failed: {e}")
        a_sample = np.zeros((max_order + 1, max_order + 1), dtype=complex)
    return a_sample

def extract_a_moments_uncertainty_parallel(S_moments, S_uncertainties, h_moments, max_order, num_samples=100, n_jobs=-1):
    """
    Propagate uncertainties from S_moments to the extracted signal moments a_moments
    using a Monte Carlo sampling approach in parallel.
    
    For each Monte Carlo iteration, a new S_moments_sample is generated by sampling
    from a Gaussian distribution with mean=S_moments and std=S_uncertainties. The extraction
    function is applied to obtain a sample of a_moments.
    
    Parameters:
      S_moments (np.ndarray): Measured moments array with shape (max_order+1, max_order+1).
      S_uncertainties (np.ndarray): Standard deviations for S_moments (same shape).
      h_moments (dict): Dictionary with keys (n, m) for noise moments.
      max_order (int): Maximum order considered.
      num_samples (int): Number of Monte Carlo samples.
      n_jobs (int): Number of parallel jobs (default: -1 uses all available cores).
      
    Returns:
      a_mean (np.ndarray): Mean extracted signal moments (shape (max_order+1, max_order+1)).
      a_std (np.ndarray): Standard deviation (uncertainty) of extracted signal moments (same shape).
    """
    # Generate samples in parallel
    samples = Parallel(n_jobs=n_jobs)(
        delayed(sample_a_moments)(S_moments, S_uncertainties, h_moments, max_order)
        for _ in range(num_samples)
    )
    samples = np.array(samples)  # shape: (num_samples, max_order+1, max_order+1)
    a_mean = np.mean(samples, axis=0)
    a_std = np.std(samples, axis=0)
    return a_mean, a_std


def annihilation_operator(N):
    """Creates the annihilation operator a in a Fock space of dimension N."""
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a

def precompute_moment_operators(N, max_order):
    """
    Precompute the operators E_{n,m} = (a†)^n a^m for n,m = 0,...,max_order 
    in a truncated Fock space of dimension N.
    """
    a = annihilation_operator(N)
    a_dag = a.conj().T
    E_ops = {}
    for n in range(max_order+1):
        for m in range(max_order+1):
            op_n = np.linalg.matrix_power(a_dag, n) if n > 0 else np.eye(N, dtype=complex)
            op_m = np.linalg.matrix_power(a, m) if m > 0 else np.eye(N, dtype=complex)
            E_ops[(n, m)] = op_n @ op_m
    return E_ops

def moments_to_density_matrix_ml(measured_moments, uncertainties, N):
    """
    Reconstruct the density matrix ρ from measured moments using maximum likelihood.
    
    This routine minimizes the weighted squared difference between the measured moments
    and the moments computed from ρ:
    
        L = - Σₙ,ₘ  (1/δₙ,ₘ²) * | Tr(ρ E_{n,m}) - M_{n,m} |²,
    
    where E_{n,m} = (a†)^n a^m and M_{n,m} are the measured moments.
    
    Parameters:
      measured_moments (np.ndarray): Array of measured moments with shape (M+1, M+1)
                                     (M is the maximum moment order).
      uncertainties (np.ndarray): Array of standard deviations δₙ,ₘ (same shape).
      N (int): Truncation dimension for the density matrix (Fock space dimension).
    
    Returns:
      np.ndarray: The estimated density matrix (N x N).
    """
    max_order = measured_moments.shape[0] - 1
    E_ops = precompute_moment_operators(N, max_order)
    
    # Define the density matrix variable: ρ must be Hermitian, PSD, and Tr(ρ)=1.
    rho = cvx.Variable((N, N), complex=True)
    constraints = [rho >> 0, cvx.trace(rho) == 1, rho == rho.H]
    
    objective_terms = []
    for n in range(max_order+1):
        for m in range(max_order+1):
            # Weight each term by the inverse variance (using uncertainties δₙ,ₘ)
            delta = uncertainties[n, m]
            weight = 1.0 / (delta**2) if delta > 0 else 1.0
            diff = cvx.trace(rho @ E_ops[(n, m)]) - measured_moments[n, m]
            objective_terms.append(weight * cvx.square(cvx.abs(diff)))
    
    objective = cvx.Minimize(cvx.sum(objective_terms))
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.SCS)  # Other solvers (e.g. MOSEK) may be used if available
    
    return rho.value

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