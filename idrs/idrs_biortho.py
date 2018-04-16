import numpy as np
from utils import orth
from scipy.sparse.linalg.isolve.utils import make_system


def idrs_biortho(A, b, x0=None, tol=1e-5, norm=np.linalg.norm, s=4, maxiter=None,callback=None,init_seed=None):
    """
    References
    ----------
    .. [1] P. Sonneveld and M. B. van Gijzen
             SIAM J. Sci. Comput. Vol. 31, No. 2, pp. 1035--1062, (2008).
    .. [2] M. B. van Gijzen and P. Sonneveld
             ACM Trans. Math. Software,, Vol. 38, No. 1, pp. 5:1-5:19, (2011).
    """
    
    dtype = A.dtype
    A,B,x,b,postprocess = make_system(A,None,x0,b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    if callback is not None:
        callback(x)

    r = b - A.dot(x)
    rnrm = norm(r)
    

    np.random.seed(init_seed)
    P = np.random.randn(n, s)
    P[:,0] = r # For comparison with BiCdRStab
    P = orth(P).T

    if callback is not None:
        callback(x)

    tolr = tol*norm(b)
    if rnrm < tolr:
        return postprocess(x)

    dR = np.zeros((n, s), dtype=dtype)
    dX = np.zeros((n, s), dtype=dtype)
    M = np.eye(s, dtype=dtype)
    om = 1.0
    iter_ = 0

    while rnrm >= tolr and iter_ < maxiter:
        m = P.dot(r)
        for k in range(0, s):
            c = np.linalg.solve(M[k:s, k:s], m[k:s])
            v = r - dR[:,k:s].dot(c)
            v = B.dot(v)

            dX[:,k] = dX[:,k:s].dot(c) + om*v
            dR[:,k] = A.dot(dX[:,k])

            for i in range(0, k):
                alpha = np.dot(P[i, :], dR[:, k]) / M[i, i]
                dR[:, k] = dR[:, k] -alpha*dR[:, i]
                dX[:, k] = dX[:, k] -alpha*dX[:, i]

            M[k:s, k] = np.dot(P[k:s,:], dR[:, k])

            beta = m[k] / M[k, k]
            r -= beta*dR[:, k]
            rnrm = np.linalg.norm(r)

            x += beta*dX[:, k]
            
            if callback is not None:
                callback(x)

            iter_ += 1
           

            if k+1 < s:
                m[k + 1:s] -= beta* M[k + 1:s, k]


        if rnrm < tolr or iter_ >= maxiter:
            break
       
        v = B.dot(r)
        t = A.dot(v)

        om = np.dot(t,r)/np.dot(t,t)
        
        x += om*v
        if callback is not None:
            callback(x)

        r -= om*t
        rnrm = norm(r)

        iter_ += 1

    return postprocess(x)[:,None]
