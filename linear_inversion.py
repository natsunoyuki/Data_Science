# Functions for linear inversion (linear regression).
from scipy.optimize import linprog
import numpy as np

def least_squares(d, G):
    """
    Linear inversion using least squares to get the Penrose inverse
    for over determined problems.
    For a linear system given by: |d> = G |m>
    |m_est> = Gg |d>
    Minimize the error E = <e|e> where |e> = |d> - G|m>
    Gg = [G.T G]**-1 G.T
    
    Inputs
    ------
    d: np.array
        np.array of observations
    G: np.array
        np.array of the inversion kernel
        
    Returns
    -------
    m: np.array
        np.array of the inverted model parameters
    """    
    m = np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)
    m = np.dot(m, d)
    return m

def min_length(d, G):
    """
    Linear inversion using minimum length to get the Penrose inverse
    for under determined problems.
    For a linear system given by: |d> = G |m>
    |m_est> = Gg |d>
    Minimize the function: E = <m|m> + <l|e>
    Gg = G.T [G G.T]**-1 
    
    Inputs
    ------
    d: np.array
        np.array of observations
    G: np.array
        np.array of the inversion kernel
        
    Returns
    -------
    m: np.array
        np.array of the inverted model parameters
    """        
    m = np.dot(G.T, np.linalg.inv(np.dot(G, G.T)))
    m = np.dot(m, d)
    return m

def SVD(d, G, l = 0.01):
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
    d: np.array
        np.array of observations
    G: np.array
        np.array of the inversion kernel
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

def l1_norm_inversion(d, G, sd = 1.0):
    """
    Linear inversion using L1 norm error instead of mean squared error for
    over determined problems.
    The inversion problem is transformed into a linear programming problem
    and solved using the linprog() function from scipy.optimize
    See Geophysical Data Analysis: Discrete Inverse Theory MATLAB Edition
    Third Edition by William Menke pages 153-157 for more details.
    
    Inputs
    ------
    d: np.array
        np.array of observations
    G: np.array
        np.array of the inversion kernel
    sd: float
        standard deviation. Set to 1 by default
        
    Returns
    -------
    m: np.array
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
    mls = least_squares(d, G)
    mupperbound = 10 * np.max(np.abs(mls))
    b[L:L+2*M] = mupperbound
    
    res = linprog(f, A, b, Aeq, beq)
    
    mest_l1 = res['x'][:M] - res['x'][M:2*M]
    return mest_l1
