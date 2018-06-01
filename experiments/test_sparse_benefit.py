import numpy as np
import scipy.io as scio
import scipy.sparse as scsp

RANDOM_SEED = 42
rand = np.random.RandomState(RANDOM_SEED)


A = scio.mmread('matricies/jpwh_991.mtx')
X_true = np.random.rand(A.shape[0])
b = A.dot(X_true)

def test1():
	return A.toarray().dot(b)

def test2():
	return A.dot(b)

if __name__ == '__main__':
	import timeit
	assert np.allclose(test1(),test2())
	
	t1 = timeit.repeat("test1()", repeat=10, number=100, setup="from __main__ import test1")
	t2 = timeit.repeat("test2()", repeat=10, number=100, setup="from __main__ import test2")
	print("Dense matmul time: {}".format(np.mean(t1)))
	print("Sparse matmul time: {}".format(np.mean(t2)))