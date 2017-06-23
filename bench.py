import numpy as np
import gpusolver
import itertools, time
from scipy.sparse import csr_matrix

nfreqs = np.arange(1,15,2)
nants = 350

T_transfer, T_compute, Asize = [], [], []

for nfq in nfreqs[::-1]:
	nrows = nfq*nants*(nants-1)/2
	Asize.append(nrows*nants)
	rows = np.repeat(np.arange(nrows), 2).astype(np.int32)
	cols = np.zeros_like(rows)
	data = np.ones_like(rows).astype(np.float32)
	b = np.random.random(nrows).astype(np.float32)

	i = 0
	for f in xrange(nfq):
		for pair in itertools.combinations(np.arange(nants), 2):
			cols[i], cols[i+1] = pair[0], pair[1]
			i += 2

	Acsr = csr_matrix( (data,(rows,cols)), shape=(nrows,nants) )

	solver = gpusolver.DnSolver(nrows, nants)
	t0 = time.time()
	solver.from_csr(Acsr.indptr, Acsr.indices, Acsr.data, b)
	t1 = time.time()
	solver.solve(1, 0)
	t2 = time.time()
	x = solver.retrieve()
	t3 = time.time()
	T_compute.append(t2-t1)
	T_transfer.append((t1-t0)+(t3-t2))

import matplotlib.pyplot as plt 
plt.figure()
plt.plot(Asize, T_compute, label='compute')
plt.plot(Asize, T_transfer, label='tranfer')
plt.legend()
plt.show()
#import IPython; IPython.embed()