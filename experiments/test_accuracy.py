import numpy as np
import scipy.io as scio
import scipy.sparse as scsp

from GMRES.gmres import gmres
from BiCGStab.bicgstab import bicgstab
from cg.cg import conjugate_gradient
from cg.bcg import biconjugate_gradient
from cg.bcg_stab import biconjugate_gradient_stab
from idrs.idrs import idrs
from idrs.idrs_biortho import idrs_biortho
from idrs.idrs_biortho_v2 import idrs_biortho as idrs_biortho_v2
from idrs.idr_s2 import idr_s2

RANDOM_SEED = 42
rand = np.random.RandomState(RANDOM_SEED)


M_Kryl = 10000
TOL = 1e-15
NORM = np.linalg.norm

def gen_mat(path):
	A = scio.mmread(path)
	X_true = np.random.rand(A.shape[0])
	b = A.dot(X_true)
	return A,b,X_true


A,b,x_true = gen_mat('matricies/add32.mtx')
#jpwh_991.mtx
#add32.mtx


def test_spsolve():
	return "SPsolve", scsp.linalg.spsolve(A,b)

def test_gmres():
	x0=np.zeros(b.size)
	res = gmres(A,b, x0, M_Kryl,TOL)
	return "GMRES", res[0]

def test_cg():
	x0=np.zeros(b.size)
	res = conjugate_gradient(A,b, x0, tol=TOL, maxiter=M_Kryl,norm=NORM)
	return "CG", res

def test_bcg():
	x0=np.zeros(b.size)
	res = biconjugate_gradient(A,b, x0, tol=TOL, maxiter=M_Kryl,norm=NORM)
	return "BiCG", res

def test_bcg_stab_1():
	x0=np.zeros(b.size)
	res = biconjugate_gradient_stab(A,b, x0, tol=TOL, maxiter=M_Kryl,norm=NORM)
	return "BiCGStab_1", res

def test_bcg_stab_2():
	x0=np.zeros(b.size)
	res = bicgstab(A,b, x0, M_Kryl,TOL)
	return "BiCGStab_2", res[0]

def test_idrs():
	x0=np.zeros(b.size)
	res = idrs(A,b, x0, tol=TOL,s=3, maxiter=M_Kryl,norm=NORM,init_seed=RANDOM_SEED)
	return "IDR(s)", res

def test_idrs_biortho():
	x0=np.zeros(b.size)
	res = idrs_biortho(A,b, x0, tol=TOL,s=2, maxiter=M_Kryl,norm=NORM,init_seed=RANDOM_SEED)
	return "IDR(s)BiOrtho", res

def test_idrs_biorthov2():
	x0=np.zeros(b.size)
	res = idrs_biortho_v2(A,b, x0, tol=TOL,s=2, maxiter=M_Kryl,norm=NORM,init_seed=RANDOM_SEED)
	return "IDR(s)BiOrthov2", res

def test_idr_s2():
	x0=np.zeros(b.size)
	res = idr_s2(A,b, x0, M_Kryl,TOL)
	return "IDR(2)", res[0]

if __name__ == '__main__':
	for f in [test_spsolve,
               test_gmres,
               test_cg,
               test_bcg,
               test_bcg_stab_1,
               test_bcg_stab_2,
               #test_idrs,
               #test_idr_s2,
               test_idrs_biortho,
               test_idrs_biorthov2]:
		res = []
		for i in range(3):
			name, x = f()
			res.append(NORM(x_true-np.ravel(x)))
		x = np.mean(res)
		print("{}: {}".format(name,x))
