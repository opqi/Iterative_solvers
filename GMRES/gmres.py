import numpy as np
import scipy.io as scio
import scipy.sparse as scsp

def gmres(a, b, x0, m, tol = 1e-15):
    """Solves the linear system Ax = b with the GMRES(m)
    
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
            Tolerance for the stopping criterion: rorm(impicit residual) <= tol.
    
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
    v = b - a.dot(x0)
    hh = np.linalg.norm(v)
    in_res = hh
    
    if hh <= tol:
        im_res = hh
        x = x0
        n_mult = 1
        return x, in_res, im_res, n_mult
    else:
        h = np.zeros((m,m))
        c = np.zeros((m,1))
        s = np.zeros((m,1))
        y = np.zeros((m+1,1))
        y[0] = hh
        j = 0
        im_res = hh
    
        while j < m and im_res > tol:
            
            v[j:] = v[j:]/hh
            vnew = a.dot(v[j:].T)
            h[0:j+1,j] = v.dot(vnew).T

            if(v.size == b.shape[0]):
                vnew -= v*h[0:j+1,j]
            else:
                vnew -= v.T.dot(h[0:j+1,j]).reshape((b.size,1))
            hh = np.linalg.norm(vnew)

            for i in range(0,j):
                h[i:i+2,j] = np.vstack((np.hstack((c[i],s[i])),np.hstack((-s[i],c[i])))).dot(h[i:i+2,j])

            if(hh!=0.0):
                ww=np.linalg.norm(np.hstack((h[j,j],hh)))
                c[j] = h[j,j]/ww
                s[j] = hh/ww
                y[j+1] = -s[j]*y[j]
                y[j] = c[j]*y[j]
                h[j, j] = ww
                im_res = float(abs(y[j+1]))
            else:
                im_res = 0.0

            j+=1
            v = np.vstack((v, vnew.T))
        j-=1
        y[j] = y[j]/h[j,j]
        for i in range(j-1,-1,-1):
            y[i] = (y[i] - h[i,i+1:j+1].dot(y[i+1:j+1]))/h[i,i]
        x = x0 + v[:j+1].T.dot(y[0:j+1]).T
        n_mult = j+1
        return x, in_res, im_res, n_mult