# Functions for linear inversion (linear regression).
from scipy.optimize import linprog
import numpy as np

# List of linear inversion solvers in this script:
# 1. Linear least squares inversion/regression for the over determined problem
# m = least_squares(G, d)
# 2. Linear inversion/regression for the under determined problem
# m = min_length(G, d)
# 3. Generalized SVD linear inversion/regression
# m = SVD(G, d, l = 0.01)
# 4. L1 norm linear inversion/regression for the over determined problem
# m = l1_norm_inversion(G, d, sd = 1.0)

# Comments regarding the inversion kernel G:
# The inversion kernel must be created properly for an accurate inversion result.
# We demonstrate how the inversion kernel should look like using examples.
# 1. Linear inversion of y = mx + c
# where x = [0.1, 0.2, 0.3, 0.4, 0.5]:
#     0.1 1
#     0.2 1
# G = 0.3 1
#     0.4 1
#     0.5 1
# 2. Linear inversion of y = ax**2 + bx + c
# where x = [0.1, 0.2, 0.3, 0.4, 0.5]:
#     0.01 0.1 1
#     0.04 0.2 1
# G = 0.09 0.3 1
#     0.16 0.4 1
#     0.25 0.5 1
# By convention, each row of G corresponds to one data point, and each column of G corresponds
# to the coefficients of a particular model parameter, such as m and c in the 1st case.

def least_squares(G, d):
    """
    Linear inversion using least squares to get the Penrose inverse
    for over determined problems.
    For a linear system given by: |d> = G |m>
    |m_est> = Gg |d>
    Minimize the error E = <e|e> where |e> = |d> - G|m>
    Gg = [G.T G]**-1 G.T
    
    Inputs
    ------
    G: np.array
        np.array of the inversion kernel. Equivalent to X in scikit-learn 
    d: np.array
        np.array of observations. Equivalent to y in scikit-learn
        
    Returns
    -------
    m: np.array
        np.array of the inverted model parameters
    """    
    m = np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)
    m = np.dot(m, d)
    return m

def min_length(G, d):
    """
    Linear inversion using minimum length to get the Penrose inverse
    for under determined problems.
    For a linear system given by: |d> = G |m>
    |m_est> = Gg |d>
    Minimize the function: E = <m|m> + <l|e>
    Gg = G.T [G G.T]**-1 
    
    Inputs
    ------
    G: np.array
        np.array of the inversion kernel. Equivalent to X in scikit-learn 
    d: np.array
        np.array of observations. Equivalent to y in scikit-learn
        
    Returns
    -------
    m: np.array
        np.array of the inverted model parameters
    """        
    m = np.dot(G.T, np.linalg.inv(np.dot(G, G.T)))
    m = np.dot(m, d)
    return m

def SVD(G, d, l = 0.01):
    """
    Linear inversion using SVD to get the Penrose inverse
    For a linear system given by: |d> = G |m>
    |m_est> = Gg |d>
    The SVD of some matrix G is given by: G = U S V
    Take only the largest eigenvalues of S, and limit the number of columns
    of U and rows of V 
    We can then obtain the Penrose inverse using the limited V, S and U.
    
    Inputs
    ------
    G: np.array
        np.array of the inversion kernel. Equivalent to X in scikit-learn 
    d: np.array
        np.array of observations. Equivalent to y in scikit-learn
    l: float
        upper limit of the tolerance level of the SVD eigenvalues to treat as 0
        any eigenvalues smaller than l will be treated as 0
        
    Returns
    -------
    m: np.array
        np.array of the inverted model parameters
    """
    u, s, vh = np.linalg.svd(G)
    cond = s > (np.max(s) * l) 
    s = s[:sum(cond)]
    u = u[:,:sum(cond)]
    vh = vh[:sum(cond),:]
    Gg = np.dot(vh.T, np.linalg.inv(np.diag(s)))
    Gg = np.dot(Gg, u.T)
    m = np.dot(Gg, d)
    return m

def l1_norm_inversion(G, d, sd = 1.0):
    """
    Linear inversion using L1 norm error instead of mean squared error for
    over determined problems.
    The inversion problem is transformed into a linear programming problem
    and solved using the linprog() function from scipy.optimize
    See Geophysical Data Analysis: Discrete Inverse Theory MATLAB Edition
    Third Edition by William Menke pages 153-157 for more details.
    
    Inputs
    ------
    G: np.array
        np.array of the inversion kernel. Equivalent to X in scikit-learn 
    d: np.array
        np.array of observations. Equivalent to y in scikit-learn
    sd: float
        variance of the measurement d. Set to 1 by default
        
    Returns
    -------
    mest_l1: np.array
        np.array of the inverted model parameters
    """
    N, M = np.shape(G)
    L = 2 * M + 3 * N
    f = np.zeros(L)
    f[2*M:2*M+N] = 1 / sd

    Aeq = np.zeros([2*N, L])
    beq = np.zeros(2*N)
    
    Aeq[:N, :M] = G
    Aeq[:N, M:2*M] = -G
    Aeq[:N, 2*M:2*M+N] = -np.eye(N)
    Aeq[:N, 2*M+N:2*M+2*N] = np.eye(N)
    beq[:N] = d
    
    Aeq[N:2*N, :M] = G
    Aeq[N:2*N, M:2*M] = -G
    Aeq[N:2*N, 2*M:2*M+N] = np.eye(N)
    Aeq[N:2*N, 2*M+2*N:2*M+3*N] = -np.eye(N)
    beq[N:2*N] = d
    
    A = np.zeros([L+2*M, L])
    b = np.zeros(L+2*M)
    A[:L, :] = -np.eye(L)
    b[:L] = np.zeros(L)
    
    A[L:L+2*M] = np.eye(2*M, L)
    mls = least_squares(G, d)
    mupperbound = 10 * np.max(np.abs(mls))
    b[L:L+2*M] = mupperbound
    
    res = linprog(f, A, b, Aeq, beq)
    
    mest_l1 = res['x'][:M] - res['x'][M:2*M]
    return mest_l1
