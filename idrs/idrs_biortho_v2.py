import numpy as np
from .utils import orth
from scipy.sparse.linalg.isolve.utils import make_system


def idrs_biortho(A, b, x0=None, tol=1e-5, norm=np.linalg.norm, s=2, maxiter=None, B=None, callback=None,init_seed=None):
    """
    Induced Dimension Reduction method [IDR(s)] to solve A x = b
    Parameters
    ----------
    A : sparse matrix, dense matrix or LinearOperator
        The real or complex n-by-n matrix of the linear system.
    b : array or matrix
        Right hand side of the linear system; Shape (N,) or (N,1).
    x0  : array or matrix
        Initial guess.
    tol : float
        Tolerance to achieve. The algorithm terminates when either the relative
        or the absolute residual is below `tol`. Default tol=1e-5
    norm : function
        Matrix or vector norm. 
    s : integer
        specifies the dimension of the shadow space. Normally, a higher
        s gives faster convergence, but also makes the method more expensive.
        Default s=2.
    maxiter : integer, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    B : sparse matrix, dense matrix or LinearOperator
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    Returns
    -------
    x : array or matrix
        The converged solution.
    info : integer
        Provides convergence information:
            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown
    """
    
    dtype = A.dtype
    A,B,x,b,postprocess = make_system(A,B,x0,b)
    
    pt_r = []
    n = len(b)
    if maxiter is None:
        maxiter = round(n+n/s)

    if callback is not None:
        callback(x)

    r = b - A.dot(x)
    rnrm = norm(r)
    r_start = rnrm
    info = 0
    

    np.random.seed(init_seed)
    P = np.random.randn(n, s)
    P[:,0] = r # For comparison with BiCdRStab
    P = orth(P).T
    #P = P.T
    if callback is not None:
        callback(x)

    tolr = tol*norm(b)
    if rnrm < tolr:
        return np.array([info, rnrm/r_start, norm(b-A.dot(x))/rnrm, 0],dtype=float), np.asarray(pt_r)#postprocess(x)

    dR = np.zeros((n, s), dtype=dtype)
    dX = np.zeros((n, s), dtype=dtype)
    M = np.eye(s, dtype=dtype)
    om = 1.0
    iter_ = 0
    angle = 0.7

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
            
            #print(M[k:s, k])
            if M[k, k] == 0.0:
                # Breakdown
                return postprocess(x)[:,None]


            beta = m[k] / M[k, k]
            r -= beta*dR[:, k]
            rnrm = np.linalg.norm(r)
            pt_r.append(rnrm/r_start)

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

        om = np.dot(t,r)/(norm(t)*norm(t))#np.dot(t,t)
        rho = abs(np.dot(t,r)/(norm(t)*norm(v)))

        if rho == 0:
            return postprocess(x_ch)[:,None]
        
        if rho < angle:
            om = om * angle / rho
        
        x += om*v
        x_ch = np.copy(x)
        if callback is not None:
            callback(x)

        r -= om*t
        rnrm = norm(r)
        rnrm_ch = np.copy(rnrm)
        pt_r.append(rnrm/r_start)

        iter_ += 1
    if rnrm >= tolr:
        info = iter_

    return postprocess(x)[:,None]