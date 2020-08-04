# Linear inversion (linear regression) using linear algebra

import numpy as np

def least_squares(d, G):
    """
    Linear inversion using least squares to get the Penrose inverse
    For a linear system given by: d = G m
    m_est = Gg d
    Gg = [G.T G]**-1 G.T
    """    
    m = np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)
    m = np.dot(m, d)
    return m

def SVD(d, G):
    """
    Linear inversion using SVD to get the Penrose inverse
    For a linear system given by: d = G m
    m_est = Gg d
    The SVD of some matrix G is given by: G = U S V
    Take only the largest eigenvalues of S, and limit the number of columns
    of U and rows of V 
    We can then obtain the Penrose inverse using the limited V, S and U.
    """
    u, s, vh = np.linalg.svd(G)
    cond = s > (np.max(s) * 0.01) 
    s = s[:len(cond)]
    u = u[:,:len(cond)]
    vh = vh[:len(cond),:]
    Gg = np.dot(vh.T, np.linalg.inv(np.diag(s)))
    Gg = np.dot(Gg, u.T)
    m = np.dot(Gg, d)
    return m