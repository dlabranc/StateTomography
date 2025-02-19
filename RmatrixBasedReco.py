import numpy as np
import math
from scipy.special import factorial, eval_genlaguerre
import qutip as qt

def prepare_histograms(measurements, nbins):
    """
    Bin 2D complex measurement data into a 2D histogram and return normalized frequencies.
    
    Args:
        measurements: 1D array of complex measurement outcomes.
        n_bins: number of bins for real and imaginary histogram.        
    Returns:
        counts: 2D array of raw counts.
        alphas: value of the complex measurement corresponding to each POVM.
    """
    max_X = np.max(np.concatenate([np.abs(np.real(measurements)), np.abs(np.imag(measurements))]))
    # Define bin edges for real and imaginary parts
    real_bins = np.linspace(-max_X, max_X, nbins)  # bins over the real axis
    imag_bins = np.linspace(-max_X, max_X, nbins)  # bins over the imaginary axis

    # Get the 2D histogram frequencies and bin edges
    counts, xedges, yedges = histogram_frequencies_2d(measurements, real_bins, imag_bins)

    # Compute midpoints for each bin (X, Y grid)
    X, Y = bin_midpoints_2d(xedges, yedges)

    # Compute coherent state alphas
    alphas = (X + 1j * Y).flatten()

    return counts, alphas

def histogram_frequencies_2d(measurements, real_bins, imag_bins):
    """
    Bin 2D complex measurement data into a 2D histogram and return normalized frequencies.
    
    Args:
        measurements: 1D array of complex measurement outcomes.
        real_bins: 1D array of bin edges for the real part.
        imag_bins: 1D array of bin edges for the imaginary part.
        
    Returns:
        freqs: 2D array of normalized counts.
        counts: 2D array of raw counts.
    """
    counts, xedges, yedges = np.histogram2d(measurements.real, measurements.imag, bins=[real_bins, imag_bins])
     
    return counts, xedges, yedges

def bin_midpoints_2d(xedges, yedges):
    """
    Compute the midpoints of 2D histogram bins.
    
    Args:
        xedges: 1D array of bin edges for the real axis.
        yedges: 1D array of bin edges for the imaginary axis.
        
    Returns:
        X, Y: 2D arrays (meshgrid) of the midpoints.
    """
    x_mid = 0.5 * (xedges[:-1] + xedges[1:])
    y_mid = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(x_mid, y_mid, indexing='ij')
    return X, Y

def rhorho_reconstruction(povm_list, counts, init_rho=None, max_iter=1000, tol=1, useG=False, dilution=None,
                           improvement_tol=1e-8, patience=10, track_likelihood=False, rho_exact=None):
    """
    Perform iterative RρR reconstruction with an additional convergence check for negligible improvement.

    The likelihood for a given state ρ is defined as
        L = ∏_i [Tr(ρ Π_i)]^(f_i)
    so that the log-likelihood becomes
        logL = ∑_i f_i * log(Tr(ρ Π_i)),
    where f_i are the relative frequencies and Π_i the POVM elements.

    Args:
        povm_list: list of numpy arrays representing the POVM elements Π_i.
        counts: list or array of measured absolute frequencies N_i (should sum to N_counts).
        init_rho: initial guess for the density matrix (if None, use maximally mixed state).
        max_iter: maximum number of iterations.
        tol: convergence tolerance with respect to the value of the maximum eigenvalue of R (see Scott Glancy's paper).
        dilution: if provided, applies a dilution step in the update.
        improvement_tol: tolerance for the change in the convergence metric (|maxEig(R)-1|) between iterations.
        patience: number of consecutive iterations with negligible improvement to allow before stopping.
    
    Returns:
        rho: the reconstructed density matrix.
    """

    frequencies = counts / np.sum(counts)
    # Determine Hilbert space dimension from first POVM element.
    d = povm_list[0].shape[0]

    G = np.sum(np.array(povm_list), axis=0)
    invG = np.linalg.inv(G)
    if init_rho is None:
        rho = np.eye(d, dtype=complex) / d
    else:
        rho = init_rho.copy()
    
    # Initialize variables for additional convergence check
    prev_metric = None
    no_improvement_count = 0
    likelihood_history = [] if track_likelihood else None
    r_eigenvalue_history = [] if track_likelihood else None
    fidelity_history = [] if track_likelihood else None

    for iteration in range(max_iter):
        
        # Compute predicted probabilities p_i = Tr(ρ Π_i)
        p = np.array([np.trace(rho @ Pi).real for Pi in povm_list])
        # Avoid division by zero – add a tiny number if necessary.
        p = np.maximum(p, 1e-12)

        # Compute and record the log-likelihood: logL = Σ_i N_i * log(p_i)
        if track_likelihood:
            logL = np.sum(counts/np.sum(counts) * np.log(p))
            likelihood_history.append(logL)
        else:
            logL = 0  # not used if tracking is disabled
        
        # Construct the operator R = Σ_i (f_i/p_i) Π_i
        R = np.zeros((d, d), dtype=complex)
        for fi, pi, Pi in zip(frequencies, p, povm_list):
            R += (fi / pi) * Pi

        # Update: ρ_new = R ρ R / Tr(R ρ R)

        # If useG is True, perform the update with G instead of R.
        if useG:
            if dilution:
            # If dilution is provided, perform a diluted update.
                update_left = (np.eye(d) + dilution * invG @ R) / (1 + dilution)
                update_right = (np.eye(d) + dilution * R @ invG) / (1 + dilution)
                rho_new = update_left @ rho @ update_right
            else:
                rho_new = invG @ R @ rho @ R @ invG
        else:
            if dilution:
            # If dilution is provided, perform a diluted update.
                update_left = (np.eye(d) + dilution *  R) / (1 + dilution)
                update_right = (np.eye(d) + dilution * R) / (1 + dilution)
                rho_new = update_left @ rho @ update_right
            else:
                rho_new = R @ rho @ R 

        

        rho_new = rho_new / np.trace(rho_new)
        if rho_exact is not None:
            step_fidelity = qt.fidelity(qt.Qobj(rho_new), qt.Qobj(rho_exact))**2
        else:
            step_fidelity = None
        # Calculate maximum eigenvalue of R
        maxEigR = np.max(np.linalg.eigvals(R))
        if track_likelihood:
            r_eigenvalue_history.append(maxEigR)
            if rho_exact is not None:
                fidelity_history.append(step_fidelity)
        
        # Define the convergence metric
        current_metric = np.abs(maxEigR - 1)
        rk = np.sqrt((len(povm_list)**2 - 1) / 2) / np.sum(counts)
        
        print(f'Iteration: {iteration}/{max_iter} | '
              f'Percentage: {tol * rk * 100 / current_metric:.2f}% | '
              f'|maxEig(R)-1|: {current_metric:.2e} > {tol * rk:.2e} | '
              f'Log-Likelihood: {logL:.4f} | '
              f'Fidelity: {step_fidelity*100:.2f}%', end='\r')
        
        # Check convergence over maxEigR
        if current_metric < tol * rk:
            print()
            print(f'Converged in {iteration + 1} iterations (|maxEig(R)-1| = {current_metric:.2e}).')
            if track_likelihood:
                return rho_new, likelihood_history, r_eigenvalue_history, fidelity_history
            return rho_new

        # Additional convergence check: if the improvement is negligible over several iterations, stop.
        if prev_metric is not None:
            if np.abs(current_metric - prev_metric) < improvement_tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
        prev_metric = current_metric

        if no_improvement_count >= patience:
            print()
            print(f'No significant improvement in the last {patience} iterations. Stopping iterations.')
            if track_likelihood:
                return rho_new, likelihood_history, r_eigenvalue_history, fidelity_history
            return rho_new
        
        rho = rho_new

    print()
    print('Maximum iterations reached without full convergence.')
    if track_likelihood:
        return rho_new, likelihood_history, r_eigenvalue_history, fidelity_history
    return rho

def coherent_state_vector(alpha, d):
    """
    Compute the coherent state |alpha> in a truncated Fock basis of dimension d.
    |alpha> = exp(-|alpha|^2/2) sum_{n=0}^{d-1} (alpha^n/sqrt(n!)) |n>
    Returns a column vector.
    """
    n = np.arange(d)
    vec = np.exp(-abs(alpha)**2 / 2) * (alpha**n) / np.sqrt(factorial(n))
    return vec.reshape((d, 1))

def coherent_state_povm(alpha, d):
    """
    Return the POVM element corresponding to the coherent state |alpha>.
    We use the standard normalization: Π(α) = (1/π)|alpha><alpha|.
    """
    vec = coherent_state_vector(alpha, d)
    return (1/np.pi) * (vec @ vec.conj().T)

def displacement_operator(alpha, d):
    """
    Compute the displacement operator D(alpha) in the truncated Fock basis of dimension d.
    The matrix elements are given by:
      ⟨m|D(α)|n⟩ = exp(-|α|^2/2) * 
           {  sqrt(n!/m!) (α)^(m-n) L_m^(n-m)(|α|^2)   if m <= n,
              sqrt(m!/n!) (-α*)^(n-m) L_n^(m-n)(|α|^2)   if m > n. }
    """
    D = np.zeros((d, d), dtype=complex)
    for m in range(d):
        for n in range(d):
            if m <= n:
                # m <= n case
                D[m, n] = (np.exp(-abs(alpha)**2 / 2) *
                           (alpha**(n - m)) *
                           np.sqrt(math.factorial(m) / math.factorial(n)) *
                           eval_genlaguerre(m, n - m, abs(alpha)**2))
            else:
                # m > n case
                D[m, n] = (np.exp(-abs(alpha)**2 / 2) *
                           ((-np.conj(alpha))**(m - n)) *
                           np.sqrt(math.factorial(n) / math.factorial(m)) *
                           eval_genlaguerre(n, m - n, abs(alpha)**2))
    return D

def displaced_thermal_povm(alpha, nbar, d):
    """
    Return the POVM element corresponding to a displaced thermal state.
    
    The thermal state in the Fock basis is given by:
      ρ_th = Σ_n [ (nbar^n)/(nbar+1)^(n+1) ] |n><n|
      
    The displaced state is ρ(α) = D(α) ρ_th D(α)†.
    We normalize the POVM as (1/π) ρ(α).
    
    Args:
        alpha: displacement parameter (complex number).
        nbar: mean photon number of the thermal state.
        d: dimension of the truncated Hilbert space.
    """
    # Build thermal state in Fock basis:
    diag = np.array([ (nbar**n) / ((nbar+1)**(n+1)) for n in range(d) ])
    rho_th = np.diag(diag)
    # Displacement operator:
    D = displacement_operator(alpha, d)
    # Displaced thermal state:
    rho_disp = D @ rho_th @ D.conj().T
    return (1/np.pi) * rho_disp

def displaced_povm(alpha, rho_h):
    """
    Return the POVM element corresponding to a displaced thermal state.
    
    The thermal state in the Fock basis is given by:
      ρ_th = Σ_n [ (nbar^n)/(nbar+1)^(n+1) ] |n><n|
      
    The displaced state is ρ(α) = D(α) ρ_th D(α)†.
    We normalize the POVM as (1/π) ρ(α).
    
    Args:
        alpha: displacement parameter (complex number).
        nbar: mean photon number of the thermal state.
        d: dimension of the truncated Hilbert space.
        rho_h: detector noise reconstructed state. 
    """
    d = rho_h.shape[0]
    # Displacement operator:
    D = displacement_operator(alpha, d)
    # Displaced thermal state:
    rho_disp = D @ rho_h @ D.conj().T
    return (1/np.pi) * rho_disp