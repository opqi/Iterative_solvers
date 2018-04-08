import numpy as np
from utils import orth
from scipy.sparse.linalg.isolve.utils import make_system


def cg(A, b, x0=None, tol=1e-5, norm=np.linalg.norm, maxiter=None,callback=None,init_seed=None):

    dtype = A.dtype
    A,B,x,b,postprocess = make_system(A,None,x0,b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    r = b - A.dot(x)
    rnrm = norm(r)
    p = r

    if callback is not None:
        callback(x)

    tolr = tol*norm(b)
    if rnrm < tolr:
        return postprocess(x)

    for i in range(maxiter):
        y = r.T.dot(r)
        z = A.dot(p)
        alpha = y/(z.T.dot(p))
        x += alpha*p
        
        if callback is not None:
            callback(x)
        
        r -= alpha*z
        rnrm = norm(r)
        if rnrm < tolr:
            break

        p =  r + r.T.dot(r)/y * p

    return postprocess(x)
