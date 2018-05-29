import numpy as np
from .utils import orth
from scipy.sparse.linalg.isolve.utils import make_system


def idrs(A, b, x0=None, tol=1e-5, norm=np.linalg.norm, s=4, maxiter=None,callback=None,init_seed=None):
    """
    References
    ----------
    .. [1] P. Sonneveld and M. B. van Gijzen
             SIAM J. Sci. Comput. Vol. 31, No. 2, pp. 1035--1062, (2008).
    .. [2] M. B. van Gijzen and P. Sonneveld
             ACM Trans. Math. Software,, Vol. 38, No. 1, pp. 5:1-5:19, (2011).
    """

    dtype = A.dtype
    A,_,x,b,postprocess = make_system(A,None,x0,b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    if callback is not None:
        callback(x)

    r = b - A.dot(x)

    np.random.seed(init_seed)
    P = np.random.randn(n, s)
    P[:,0] = r # For comparison with BiCGStab
    P = orth(P).T

    rnrm = norm(r)
    # Relative tolerance
    tolr = tol*norm(b)
    if rnrm < tolr:
        # Initial guess is a good enough solution
        return postprocess(x)

    # Initialization
    dR = np.zeros((n, s), dtype=dtype)
    dX = np.zeros_like(dR)
    M = np.zeros((s, s), dtype=dtype)
    for i in range(s):
        v = A.dot(r)
        om = np.dot(v,r)/np.dot(v,v)
        dX[:,i] = om*r
        dR[:,i] = -om*v
        x += dX[:,i]
        
        if callback is not None:
            callback(x)

        r += dR[:,i]
        M[:,i] = P.dot(dR[:,i])
            

    iter_ = s
    current = 0
    m = P.dot(r)
    while rnrm > tolr  and iter_ < maxiter :
        for k in range(s+1):
            c = np.linalg.solve(M, m)
            q = -dR.dot(c)
            v = r + q
            if k==0:
                t = A.dot(v)
                om = np.dot(t,v)/np.dot(v,v)
                dR[:,current] = q - np.dot(om,t)
                dX[:,current] = -dX.dot(c) + om*v
            else:
                dX[:,current] = -dX.dot(c) + om*v
                dR[:,current] = -A.dot(dX[:,current])
            x += dX[:,current]

            if callback is not None:
                callback(x)

            r += dR[:,current]
            iter_ += 1
            dm = P.dot(dR[:,current])
            M[:,current] = dm
            m += dm
        current = (current+1)%s

        rnrm = norm(r)

    return postprocess(x)[:,None]
