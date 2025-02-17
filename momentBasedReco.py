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

def calculate_moments_uncertainty_vectorized(S_measurements, max_order):
    """
    Vectorized calculation of moments of S from the given measurements and their standard deviations.

    Parameters:
      S_measurements (np.ndarray): Array of S measurements (shape: num_samples,).
      max_order (int): The maximum order of moments to calculate.

    Returns:
      moments (np.ndarray): Array of moments of shape (max_order+1, max_order+1).
      uncertainties (np.ndarray): Array of standard deviations for each moment (same shape).
    """
    num_samples = S_measurements.shape[0]
    M = max_order + 1
    # Create an array of exponents [0, 1, ..., max_order]
    exponents = np.arange(M)
    # Compute S_measurements raised to each power:
    # X_powers has shape (num_samples, M) where X_powers[i, m] = (S_measurements[i])^m.
    X_powers = S_measurements[:, np.newaxis] ** exponents  
    # Similarly for the complex conjugate.
    conjX_powers = np.conjugate(S_measurements)[:, np.newaxis] ** exponents  
    
    # For each measurement i, compute the outer product to get the (n,m) moment sample:
    # moment_samples has shape (num_samples, M, M) with element:
    # moment_samples[i, n, m] = conjX_powers[i, n] * X_powers[i, m].
    moment_samples = conjX_powers[:, :, np.newaxis] * X_powers[:, np.newaxis, :]  # shape: (num_samples, M, M)
    
    # Compute the mean moments over all measurements:
    mean_moments = np.mean(moment_samples, axis=0)  # shape: (M, M)
    # As in the original code, we take the absolute value of the mean moment.
    moments = np.abs(mean_moments)
    
    # Compute the uncertainty (standard deviation) for each moment:
    # For each (n,m), compute the standard deviation of moment_samples[:, n, m].
    uncertainties = np.sqrt(np.mean(np.abs(moment_samples - mean_moments)**2, axis=0))
    
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

def build_extraction_matrix(h_moments, max_order):
    """
    Build the linear matrix A that relates the unknown signal moments a_moments
    to the measured moments S_moments via:
    
      S(n,m) = sum_{i=0}^{n} sum_{j=0}^{m} binom(n,i)*binom(m,j)* a(i,j)* h(n-i, m-j)
    
    Parameters:
      h_moments (dict): Noise moments with keys (n, m) for n,m = 0,...,max_order.
      max_order (int): Maximum order (so M = max_order+1).
      
    Returns:
      A (np.ndarray): A matrix of shape (M², M²) such that y = A x,
                      where x and y are the flattened versions of a_moments and S_moments.
    """
    M = max_order + 1
    A = np.zeros((M*M, M*M))
    # Row corresponds to equation for S(n,m), column to unknown a(i,j)
    for n in range(M):
        for m in range(M):
            row = n * M + m
            for i in range(M):
                for j in range(M):
                    col = i * M + j
                    if i <= n and j <= m:
                        A[row, col] = comb(n, i) * comb(m, j) * h_moments[n-i, m-j]
                    else:
                        A[row, col] = 0.0
    return A

def vectorized_extract_a_moments(S_samples, h_moments, max_order):
    """
    Vectorized extraction of signal moments from measured moments.
    
    For each measured S_moments sample (of shape (max_order+1, max_order+1)),
    we assume a linear relation y = A x where x are the unknown signal moments
    (flattened) and y are the measured moments (flattened). We precompute A's pseudoinverse
    and then apply it to all samples.
    
    Parameters:
      S_samples (np.ndarray): Array of shape (num_samples, M, M) of measured moments.
      h_moments (dict): Dictionary of noise moments for keys (n, m) with 0 ≤ n,m ≤ max_order.
      max_order (int): Maximum order (M = max_order+1).
      
    Returns:
      a_samples (np.ndarray): Extracted signal moments for each sample, of shape (num_samples, M, M).
    """
    num_samples, M, M2 = S_samples.shape
    if M != (max_order+1) or M2 != (max_order+1):
        raise ValueError("S_samples shape does not match max_order")
        
    # Build extraction matrix A and compute its pseudoinverse
    A = build_extraction_matrix(h_moments, max_order)
    A_pinv = np.linalg.pinv(A)
    
    # Reshape S_samples to (num_samples, M²)
    S_flat = S_samples.reshape(num_samples, -1)
    # For each sample, compute a_vec = A_pinv @ S_flat (vectorized over samples)
    a_flat = S_flat.dot(A_pinv.T)  # shape: (num_samples, M²)
    # Reshape each sample back to (M, M)
    a_samples = a_flat.reshape(num_samples, M, M)
    return a_samples

def extract_a_moments_uncertainty_vectorized(S_moments, S_uncertainties, h_moments, max_order, num_samples=100):
    """
    Propagate uncertainties from S_moments to extracted signal moments a_moments
    using a fully vectorized Monte Carlo sampling approach with NumPy.
    
    For each Monte Carlo sample, we generate a new S_moments_sample by sampling each
    element from a Gaussian with the given mean and uncertainty, then we extract a_moments
    via the linear inversion, vectorized over all samples.
    
    Parameters:
      S_moments (np.ndarray): Measured moments, shape (max_order+1, max_order+1).
      S_uncertainties (np.ndarray): Uncertainties (std dev), same shape.
      h_moments (dict): Dictionary of noise moments.
      max_order (int): Maximum order considered.
      num_samples (int): Number of Monte Carlo samples.
      
    Returns:
      a_mean (np.ndarray): Mean extracted signal moments, shape (max_order+1, max_order+1).
      a_std (np.ndarray): Standard deviation of extracted signal moments, same shape.
    """
    # Vectorized sampling: generate an array of shape (num_samples, M, M)
    M = max_order + 1
    S_samples = np.random.normal(loc=S_moments.real, scale=S_uncertainties.real, 
                                  size=(num_samples, M, M))
    
    # Vectorized extraction for all samples
    a_samples = vectorized_extract_a_moments(S_samples, h_moments, max_order)
    
    # Compute mean and std deviation over the sample axis
    a_mean = np.mean(a_samples, axis=0)
    a_std = np.std(a_samples, axis=0)
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


def g2_from_moments_with_uncertainty(moments, uncertainties):
    """
    Calculate the second-order correlation g^(2)(0) and its uncertainty from measured
    normally-ordered moments.

    We define:
        g^(2)(0) = <(a†)^2 a^2> / (<a† a>)^2
    and propagate uncertainty via:
        δg₂ = sqrt[ (δM22/(M11)^2)^2 + (2*M22*δM11/(M11)^3)^2 ],
    where M11 = moments[1,1], M22 = moments[2,2],
    δM11 = uncertainties[1,1] and δM22 = uncertainties[2,2].

    Parameters:
      moments (np.ndarray): Array of moments (e.g. with indices (n, m)).
      uncertainties (np.ndarray): Array of uncertainties for the moments (same shape).

    Returns:
      tuple: (g2, uncertainty) where g2 is the calculated second-order correlation and
             uncertainty is the propagated error.
    """
    M11 = moments[1,1]
    M22 = moments[2,2]
    if M11 == 0:
        return 0.0, 0.0
    delta_M11 = uncertainties[1,1]
    delta_M22 = uncertainties[2,2]
    g2 = M22 / (M11**2)
    delta_g2 = np.sqrt((delta_M22/(M11**2))**2 + (2 * M22 * delta_M11/(M11**3))**2)
    return g2, delta_g2