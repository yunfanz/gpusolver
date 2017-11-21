from __future__ import print_function
import numpy as np
from scipy.signal import fftconvolve
import gpusolver
import itertools, time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt 

def create_matrix(autocorr):
	mat = np.array([np.roll(autocorr, i) for i in xrange(len(autocorr))])
	uidx = np.triu_indices(len(autocorr), 1)
	lidx = np.tril_indices(len(autocorr), -1)
	mat[uidx] = np.conj(mat[uidx])
	mat[lidx] = np.conj(mat.T[lidx])
	return mat

if __name__ == "__main__":
	sig = np.load('/data1/KLT/train_snrm5/raw/0.npy')
	print(sig.dtype, sig.size)
	verify = False
	corr_len = 4096
	start = time.time()
	autocorr = fftconvolve(sig, np.conj(sig[::-1]))
	autocorr = autocorr[len(autocorr)//2:len(autocorr)//2+corr_len]
	# autocorr = np.ones(corr_len, dtype=np.complex64)
	# autocorr[0] = 1.
	solver = gpusolver.DnSolver(np.int32(corr_len))
	t1 = time.time()
	for i in xrange(10):
		print(i)
		solver.corr_from_vec(autocorr)
		print(i)
		solver.solve()
	x = solver.retrieve()
	t2 = time.time()
	if verify:
		M = create_matrix(autocorr)
		xnp = np.linalg.eigvals(M)
	print('fftconvolve time {}'.format(t1-start))
	print('solving time {}'.format((t2-t1)/10))
	import IPython; IPython.embed()

