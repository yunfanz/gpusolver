import numpy as np
import gpusolverSp as gpusolver
import itertools, time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt 

if False:
	shape = (60000,6500) 
	A = np.random.random(shape).astype(np.float32)
	b = np.random.random(shape[0]).astype(np.float32)
	T_transfer, T_compute = [], []
	Sp = np.arange(0.95,1,1)
	solver = gpusolver.DnSolver(*shape)
	for sparsity in Sp:	
		data = A * (A > sparsity)

		Acsr = csr_matrix(data)

		
		t0 = time.time()
		solver.from_csr(Acsr.indptr, Acsr.indices, Acsr.data, b)
		t1 = time.time()
		solver.solve(0)
		t2 = time.time()
		x = solver.retrieve()
		t3 = time.time()
		T_compute.append(t2-t1)
		T_transfer.append((t1-t0)+(t3-t2))
	print T_compute, T_transfer


	plt.figure()
	plt.plot(Sp, T_compute, label='compute')
	plt.plot(Sp, T_transfer, label='tranfer')
	plt.xlabel('Sparsity')
	plt.ylabel('Seconds')
	plt.legend()
	plt.savefig('bench1.png')


# Radio astronomy bench
nfreqs = np.arange(1,5,2)
nants = 350
nvis = 3000
ncols = nants+nvis

T_transfer, T_compute, Asize = [], [], []

for nfq in nfreqs:
	nrows = nfq*nants*(nants-1)/2
	Asize.append(nrows*nants)
	rows = np.repeat(np.arange(nrows), 3).astype(np.int32)
	cols = np.zeros_like(rows)
	data = np.ones_like(rows).astype(np.float32)
	b = np.random.random(nrows).astype(np.float32)

	i = 0
	for f in xrange(nfq):
		for pair in itertools.combinations(np.arange(nants), 2):
			cols[i], cols[i+1] = pair[0], pair[1]
			cols[i+2] = np.random.choice(nvis) + nants
			i += 3

	Acsr = csr_matrix( (data,(rows,cols)), shape=(nrows,ncols) )

	solver = gpusolver.DnSolver(nrows, ncols)
	t0 = time.time()
	solver.from_csr(Acsr.indptr, Acsr.indices, Acsr.data, b)
	t1 = time.time()
	for repeat in xrange(1):
		solver.solve(0)
	t2 = time.time()
	x = solver.retrieve()
	t3 = time.time()
	T_compute.append((t2-t1)/1.)
	T_transfer.append((t1-t0)+(t3-t2))


plt.figure()
plt.plot(Asize, T_compute, label='compute')
plt.plot(Asize, T_transfer, label='tranfer')
plt.legend()
plt.xlabel('Size')
plt.ylabel('Seconds')
plt.savefig('bench2.png')
import IPython; IPython.embed()