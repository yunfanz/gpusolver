import numpy as np
import gpusolverSp as gpusolver
import itertools, time
from scipy.sparse import csr_matrix, coo_matrix
import matplotlib.pyplot as plt 
from scipy.sparse.csgraph import reverse_cuthill_mckee 

def reorder_matrix(matrix, symm=True):

    # reorder based on RCM from scipy.sparse.csgraph
    rcm_perm = reverse_cuthill_mckee(csr_matrix(matrix), symm)
    rev_perm_dict = {k : rcm_perm.tolist().index(k) for k in rcm_perm}
    perm_i = [rev_perm_dict[ii] for ii in matrix.row]
    perm_j = [rev_perm_dict[jj] for jj in matrix.col]

    new_matrix = csr_matrix(
        (matrix.data, (perm_i, perm_j)), 
        shape=matrix.shape
    )
    return new_matrix, rcm_perm

def reorder_vector(vec, rcm_perm, reverse=False):
 	if reverse:
 		rev_perm_dict = {k : rcm_perm.tolist().index(k) for k in rcm_perm}
 		return np.array([vec[rev_perm_dict[k]] for k in xrange(vec.size)])
 	else:
 		return np.array([vec[rcm_perm[k]] for k in xrange(vec.size)])

if False:
	shape = (60000,1500) 
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
nants = 100
nvis = 500
ncols = nants+nvis


T_transfer, T_compute, Asize = [], [], []


print "Preparing data"
nrows = nants*(nants-1)/2
Asize.append(nrows*nants)
rows = np.repeat(np.arange(nrows), 3).astype(np.int32)
cols = np.zeros_like(rows)
data = np.ones_like(rows).astype(np.float32)
b = np.random.random(nrows).astype(np.float32)

i = 0
for pair in itertools.combinations(np.arange(nants), 2):
	cols[i], cols[i+1] = pair[0], pair[1]
	cols[i+2] = np.random.choice(nvis) + nants
	i += 3

A = csr_matrix( (data,(rows,cols)), shape=(nrows,ncols) ).todense()
AtA = np.dot(A.T, A)
Atb_ = np.asarray(np.dot(A.T, b)).flatten()
AtAcsr, rcm_perm = reorder_matrix(coo_matrix(AtA))
Atb = reorder_vector(Atb_, rcm_perm, False)
#Atb = Atb_
#import IPython; IPython.embed()
#AtAcsr = csr_matrix(AtA)
x_np = np.asarray(np.dot(np.linalg.pinv(AtA), Atb_)).flatten()

batches = np.array([16, 32, 64, 128, 256, 512])
for batch in batches:
	#import IPython; IPython.embed()
	dataA = np.hstack([AtAcsr.data for i in xrange(batch)])
	datab = np.hstack([Atb for i in xrange(batch)])

	print "got AtA with sparsity", 1 - float(AtAcsr.nnz)/(AtAcsr.shape[0]**2)
	#import IPython; IPython.embed()
	solver = gpusolver.SpSolver(AtA.shape[0], AtA.shape[0], batch)
	solver.prepare_workspace(AtAcsr.indptr, AtAcsr.indices)
	t0 = time.time()
	solver.from_csr(dataA, datab)
	t1 = time.time()
	#x = solver.solve_Axb_and_retrieve()
	x = solver.solve_Axb_and_retrieve()[:AtA.shape[1]]
	t2 = time.time()
	T_compute.append((t2-t1))
	T_transfer.append((t1-t0))
	print x.shape, x_np.shape
	x = reorder_vector(x, rcm_perm, True)
	#import IPython; IPython.embed()
	print 'Relative Error', np.sum((np.asarray(np.dot(AtA, x))[0]-Atb_)**2)/np.sum(Atb**2)
print "compute per sample: ", np.array(T_compute)/batches
print "transfer per sample: ", T_transfer
# plt.figure()
# plt.plot(Asize, T_compute, label='compute')
# plt.plot(Asize, T_transfer, label='tranfer')
# plt.legend()
# plt.xlabel('Size')
# plt.ylabel('Seconds')
# plt.savefig('bench2.png')
#import IPython; IPython.embed()