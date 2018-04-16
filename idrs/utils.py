from numpy.linalg import svd

def orth(A):
    u, s, vh = svd(A, full_matrices=False)
    M, N = A.shape
    Q = u[:,:N]
    return Q