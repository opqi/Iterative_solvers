import numpy as np
import scipy.io as scio
import scipy.sparse as scsp
import scipy.linalg as scli

def bicgstab(a, b, x0, m, tol = 1e-15):
    """Solves the linear system Ax = b with the BiCGStab(m)
    
    Parameters
    ----------
        a : ndarray or coo_matrix
            Square real unsymmetric matrix.
        b : ndarray
            Column vector.
        x0 : ndarray
            Initial guess.
        m : int
            Maximum size of Krylov subspace, m > 1.
        tol : float
            Tolerance for the stopping criterion: norm(impicit residual) <= tol.
    
    Returns
    -------
        x : ndarray
            Approximate solution.
        in_res : ndarray
            Initial residual.
        im_res : ndarray
            Final implicit residual.
        n_mult : int
            Number of MV multiplications.
    """
    u = b - a.dot(x0)
    n_mult = 1
    hh = np.linalg.norm(u)
    in_res = hh
    if hh <= tol:
        im_res = hh
        x = np.copy(x0)
        return x, in_res, im_res, n_mult
    else:
        x = np.copy(x0)
        r = np.copy(u)
        p = np.copy(u)
        rho = hh**2
        j=0
        while j<m and hh>tol:
            v = a.dot(p)
            j+=1
            delta = u.T.dot(v)
            alpha = rho/delta
            x += p*alpha
            r -= v*alpha
            q = a.dot(r)
            j+=1
            omega=(q.T.dot(r))/(q.T.dot(q))
            x += r*omega
            r -= q*omega
            rhold = rho
            rho = u.T.dot(r)   
            hh = np.linalg.norm(r) 
            beta = (rho/rhold)*(alpha/omega)
            p = r + (p-v*omega)*beta
        im_res = hh
        n_mult += j
        return x, in_res, im_res, n_mult