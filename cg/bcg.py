import numpy as np
import scipy.sparse as scsp


def biconjugate_gradient(A, b, x0=None, tol=1e-5, norm=np.linalg.norm, maxiter=None,callback=None,init_seed=None):

    dtype = A.dtype
    n = len(b)

    if x0 is None:
        x = np.zeros(n,dtype=dtype)
    else:
        x = x0

    A = scsp.coo_matrix(A)   
    
    if maxiter is None:
        maxiter = n*10

    r = b - A.dot(x)
    r_hat = r.copy()
    rnrm = norm(r)
    
    p = r.copy()
    p_hat = r_hat.copy()
    
    if callback is not None:
        callback(x)

    tolr = tol*norm(b)
    if rnrm < tolr:
        return x

    for i in range(maxiter):
        r_norm = r_hat.T.dot(r)

        Ap = A.dot(p)
        alpha = r_norm / Ap.T.dot(p_hat)
        x += alpha*p
        
        if callback is not None:
            callback(x)
        
        r -= alpha*Ap
        r_hat -= alpha*(A.T.dot(p_hat))
        rnrm = norm(r)
        if rnrm < tolr:
            break

        beta = r_hat.T.dot(r)/r_norm
        p =  r + p*beta
        p_hat =  r_hat + p_hat*beta
        
    return x
