import time
import numpy as np
import scipy.io as scio
import scipy.sparse as scsp

from GMRES.gmres import gmres
from BiCGStab.bicgstab import bicgstab
from cg.cg import conjugate_gradient
from cg.bcg import biconjugate_gradient
from cg.bcg_stab import biconjugate_gradient_stab
from idrs.idrs_biortho import idrs_biortho
from idrs.idrs_biortho_v2 import idrs_biortho as idrs_biortho_v2

RANDOM_SEED = 42
rand = np.random.RandomState(RANDOM_SEED)


M_Kryl = 5000
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

def test_gmres(tol):
	x0=np.zeros(b.size)
	res = gmres(A,b, x0, M_Kryl,tol)
	return "GMRES", res[0]

def test_cg(tol):
	x0=np.zeros(b.size)
	res = conjugate_gradient(A,b, x0, tol=tol, maxiter=M_Kryl,norm=NORM)
	return "CG", res

def test_bcg(tol):
	x0=np.zeros(b.size)
	res = biconjugate_gradient(A,b, x0, tol=tol, maxiter=M_Kryl,norm=NORM)
	return "BiCG", res

def test_bcg_stab_1(tol):
	x0=np.zeros(b.size)
	res = biconjugate_gradient_stab(A,b, x0, tol=tol, maxiter=M_Kryl,norm=NORM)
	return "BiCGStab_1", res

def test_bcg_stab_2(tol):
	x0=np.zeros(b.size)
	res = bicgstab(A,b, x0, M_Kryl,tol)
	return "BiCGStab_2", res[0]

def test_idrs_biortho(tol):
	x0=np.zeros(b.size)
	res = idrs_biortho(A,b, x0, tol=tol,s=2, maxiter=M_Kryl,norm=NORM,init_seed=RANDOM_SEED)
	return "IDR(s)BiOrtho", res

def test_idrs_biorthov2(tol):
	x0=np.zeros(b.size)
	res = idrs_biortho_v2(A,b, x0, tol=tol,s=2, maxiter=M_Kryl,norm=NORM,init_seed=RANDOM_SEED)
	return "IDR(s)BiOrthov2", res

if __name__ == '__main__':
	TOL = np.logspace(-1,-15,num=35)

	for itr,f in enumerate([#test_spsolve,
				test_gmres,
				test_cg,
				test_bcg,
				test_bcg_stab_1,
				test_bcg_stab_2,
				#test_idrs,
				#test_idr_s2,
				test_idrs_biortho,
				test_idrs_biorthov2]):
		print("Now tested iter #", itr)
		res = []
		times = []
		for tol in TOL:
			print("\tTOL ", tol)
			r_, t_ = [], []
			for i in range(10):
				t0 = time.time()
				name, x = f(tol)
				t1 = time.time()
				r_.append(NORM(x_true-np.ravel(x))),
				t_.append(t1-t0)
				if t_[-1]>60.0 and i>0:
					break
			res.append(np.mean(r_))
			times.append(np.mean(t_))
		with open("results/{}.acc".format(name),"w") as f:
			f.write("\n".join(map(str,res)))
		with open("results/{}.time".format(name),"w") as f:
			f.write("\n".join(map(str,times)))
