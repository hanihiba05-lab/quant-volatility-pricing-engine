import numpy as np
from scipy.linalg import solve_banded

def crank_nicolson_pricing(S0, K, T, r, sigma, Smax=300, M=100, N=1000, is_american=False):
    """
    Crank-Nicolson Implicit Finite Difference Solver for Option Pricing.
    """
    dt = T / N
    dS = Smax / M
    S = np.linspace(0, Smax, M + 1)
    
    # Grid initialization
    V = np.maximum(S - K, 0) # Call payoff
    
    i = np.arange(1, M)
    alpha = 0.25 * dt * (sigma**2 * (i**2) - r * i)
    beta = -0.5 * dt * (sigma**2 * (i**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (i**2) + r * i)
    
    # Tridiagonal system setup
    A = np.zeros((3, M - 1))
    A[0, 1:] = -gamma[:-1]         # Upper diagonal
    A[1, :] = 1 - beta             # Main diagonal
    A[2, :-1] = -alpha[1:]         # Lower diagonal
    
    B = np.zeros((M - 1, M - 1))
    np.fill_diagonal(B, 1 + beta)
    np.fill_diagonal(B[1:], alpha[1:])
    np.fill_diagonal(B[:, 1:], gamma[:-1])
    
    for n in range(N):
        rhs = B @ V[1:M]
        V[1:M] = solve_banded((1, 1), A, rhs)
        if is_american:
            V[1:M] = np.maximum(V[1:M], S[1:M] - K)
            
    # Interpolate price at S0
    return np.interp(S0, S, V)