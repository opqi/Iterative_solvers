import numpy as np


def idr_s2(A, b , x0=None, tol=1e-8, m=None):
    n = len(b)
    pt_r=[]
    if m is None:
        m = np.round(n+n/2)
    if x0 is None:
        x0 = np.zeros_like(b)    
    u = b - A.dot(x0)
    nmult = 1
    hh = np.linalg.norm(u)
    rinp = hh
    pt_r.append(hh/rinp)
    if hh < tol:
        rimpl = hh
        x = x0
        return x, rimpl/rinp, np.linalg.norm(b-A.dot(x))/rimpl
    else:
        p0 = np.zeros_like(x0)
        p1 = np.copy(p0)
        s0 = np.copy(p0)
        x = np.copy(x0)
        r = u
        v = u
        a11 = 0
        a21 = 1
        omg0 = 0
        omg1 = 0
        j = 0
        while j<m and hh > tol:
            s1 = A.dot(v)
            j+=1
            if j < 3 or j == 3*np.floor(j/3):
                tau = s1.T.dot(v) / s1.T.dot(s1)
            u = p0
            p0 = p0*omg0 + p1*omg1 + v*tau
            p1 = u
            x += p0
            u = r
            r = v - s1*tau
            s1 = s0
            s0 = u - r
            hh = np.linalg.norm(r)
            pt_r.append(hh/rinp)
            if(j == 1):
                q0 = np.ones_like(u)
                q0[1::2] = 0
                q1 = np.ones_like(u)
                q1[::2] = 0
            a12 = a11
            a22 = a21
            b1 = q0.T.dot(r)
            b2 = q1.T.dot(r)
            a11 = q0.T.dot(s0)
            a21 = q1.T.dot(s0)
            delta = a11*a22 - a12*a21
            omg0 = (a22*b1 - a12*b2)/delta
            omg1 = (a11*b2 - a21*b1)/delta
            v = r - s0*omg0 - s1*omg1;
        rimpl = hh
        pt_r.append(rimpl/rinp)
        nmult += j
        return x, rimpl/rinp, np.linalg.norm(b-A.dot(x))/rimpl, nmult, np.asarray(pt_r)