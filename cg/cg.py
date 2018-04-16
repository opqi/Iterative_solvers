import numpy as np
from scipy.sparse.linalg.isolve.utils import make_system


def conjugate_gradient(A, b, x0=None, tol=1e-5, norm=np.linalg.norm, maxiter=None,callback=None,init_seed=None):

    dtype = A.dtype
    A,B,x,b,postprocess = make_system(A,None,x0,b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    r = b - A.dot(x)
    rnrm = norm(r)
    p = r.copy()

    if callback is not None:
        callback(x)

    tolr = tol*norm(b)
    if rnrm < tolr:
        return postprocess(x)[:,None]

    for i in range(maxiter):
        r_norm = r.T.dot(r)

        Ap = A.dot(p)
        alpha = r_norm / p.T.dot(Ap)
        x += alpha*p
        
        if callback is not None:
            callback(x)
        
        r -= alpha*Ap
        rnrm = norm(r)
        if rnrm < tolr:
            break

        beta = r.T.dot(r)
        p =  r + p*(beta/r_norm)
        
    return postprocess(x)[:,None]

