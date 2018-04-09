import numpy as np
import scipy.sparse as scsp


def biconjugate_gradient_stab(A, b, x0=None, tol=1e-5, norm=np.linalg.norm, maxiter=None,callback=None,init_seed=None):

    dtype = A.dtype
    n = len(b)

    if x0 is None:
        x = np.zeros((n,1),dtype=dtype)
    else:
        x = x0

    A = scsp.coo_matrix(A)   
    
    if maxiter is None:
        maxiter = n*10

    r = b - A.dot(x)
    r_hat = r.copy()
    rnrm = norm(r)
    
    p = 0
    Ap = 0
    rho = 1
    omega = 1
    alpha = 1

    if callback is not None:
        callback(x)

    tolr = tol*norm(b)
    if rnrm < tolr:
        return x

    for i in range(maxiter):
        rho_ = r_hat.T.dot(r)
        beta = (rho_/rho)*(alpha/omega)

        p =  r + beta*(p - omega*Ap)
        Ap = A.dot(p)
        alpha = rho_ / r_hat.T.dot(Ap)

        h = x + alpha*p

        if norm(b-A.dot(h)) < tolr:
            x = h
            break

        s = r - alpha*Ap
        t = A.dot(s)
        omega = t.T.dot(s)/(t.T.dot(t))

        x = h + omega*s
        r = s - omega*t

        rnrm = norm(r)
        if rnrm < tolr:
            break

        rho = rho_
        
    return x

