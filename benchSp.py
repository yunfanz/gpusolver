import numpy as np
import SpSolver as gpusolver
import itertools, time
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
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

def reorder_matrix1(matrix, symm=True):

    # reverse order on RCM from scipy.sparse.csgraph
    new_matrix = csr_matrix(matrix)
    rcm_perm = reverse_cuthill_mckee(new_matrix, symm)
    new_matrix.indices = rcm_perm.take(new_matrix.indices)
    new_matrix = csc_matrix(new_matrix)
    new_matrix.indices = rcm_perm.take(new_matrix.indices)

    new_matrix = csr_matrix(new_matrix)
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




if False:
	# Radio astronomy bench with random unique visibilities
	nants = 350
	nvis = 6000
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
	import IPython; IPython.embed()
	#AtAcsr = csr_matrix(AtA)
	#x_np = np.asarray(np.dot(np.linalg.pinv(AtA), Atb_)).flatten()

	#batches = np.array([16, 32, 64, 128, 256, 512])
	batches= np.array([1024])
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
		#print x.shape, x_np.shape
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
	import IPython; IPython.embed()



def _HERA_plotsense_dict(file, NANTS=None, keep_non_red=False):
	"""
	returns dictionaries of equivalency classes of baselines
	"""
	antpos = np.loadtxt(file)
	if NANTS is None:
		NANTS = antpos.shape[0]

	bl_gps = {}
	ant_gps = {}
	
	print "making bl_gps"
	for i in xrange(NANTS-1):
		a1pos = antpos[i]
		for j in xrange(i+1,NANTS):
			a2pos = antpos[j]
			blx, bly, blz = a2pos-a1pos
			has_entry = False
			for key in bl_gps.keys():
				#if np.hypot(key[0]-blx, key[1]-bly) < 1 or np.hypot(key[0]+blx, key[1]+bly) < 1:
				if (abs(key[0]-blx)<0.1 and abs(key[1]-bly)<0.1) or (abs(key[0]+blx)<0.1 and abs(key[1]+bly)<0.1):
					has_entry = True
					bl_gps[key] = bl_gps.get(key, []) + [(i,j)]
					break
			if not has_entry:
				bl_gps[(blx, bly)] = [(i,j)]
				
	#assert(n_unique == len(bl_gps.keys()))
	n_total = NANTS*(NANTS-1)/2


	print "making ant_gps"
	n_unique = 0
	for k, v in bl_gps.iteritems():
		if len(v) >= 2 or keep_non_red:
			for pair in v:
				ant_gps[pair] = n_unique
			n_unique += 1

	
	print "Found %d classes among %d total baselines" % (n_unique, n_total)
	return ant_gps, bl_gps




if True:
	# Radio astronomy bench from actual HERA antconfig file
	version, nants = 350, 320
	FILE = "./HERA_antconfig/antenna_positions_{}.dat".format(version)
	print "getting unique baselines"
	ant_gps, bl_gps = _HERA_plotsense_dict(FILE, nants)
	nvis = len(bl_gps.keys())
	ncols = nants + nvis
	nchan = 256

	T_transfer, T_compute = [], []
	print "Preparing data"
	nrows = nants*(nants-1)/2
	rows = np.repeat(np.arange(nrows), 3).astype(np.int32)
	cols = np.zeros_like(rows)
	data = np.ones_like(rows).astype(np.float32)
	b = np.random.random(nrows).astype(np.float32)

	i = 0
	for pair in itertools.combinations(np.arange(nants), 2):
		cols[i], cols[i+1] = pair[0], pair[1]
		try:
			cols[i+2] = ant_gps[(pair[0], pair[1])] + nants
		except(KeyError):
			pass
		i += 3

	A = csr_matrix( (data,(rows,cols)), shape=(nrows,ncols) ).todense()
	AtA = np.dot(A.T, A)
	Atb_ = np.asarray(np.dot(A.T, b)).flatten()
	AtAcsr, rcm_perm = reorder_matrix(coo_matrix(AtA))
	Atb = reorder_vector(Atb_, rcm_perm, False)
	#Atb = Atb_
	#import IPython; IPython.embed()
	#AtAcsr = csr_matrix(AtA)\
	if False:
		t0 = time.time()
		x_np = np.asarray(np.dot(np.linalg.pinv(AtA), Atb_)).flatten()
		t1 = time.time()
		print "numpy time {}".format(t1-t0)

	#batches = np.array([16, 32, 64, 128, 256, 512])
	batch = nchan #number of batches to be fed to GPU

	#stack all the data vectors together. SpSolver will figure out what is what
	#here for testing purposes all data vectors are the same
	dataA = np.hstack([AtAcsr.data for i in xrange(batch)])
	datab = np.hstack([Atb for i in xrange(batch)])

	print "got AtA with sparsity", 1 - float(AtAcsr.nnz)/(AtAcsr.shape[0]**2)


	#### initialize GPU solver ####
	solver = gpusolver.SpSolver(AtA.shape[0], AtA.shape[0], batch)
	solver.prepare_workspace(AtAcsr.indptr, AtAcsr.indices) #calculates optimal batch size from available memory
	t0 = time.time()
	solver.from_csr(dataA, datab) # feed entire dataset, may be much larger than GPU memory
	t1 = time.time()              #
	#x = solver.solve_Axb_and_retrieve()
	x = solver.solve_Axb_and_retrieve()[:AtA.shape[1]]
	t2 = time.time()
	T_compute.append((t2-t1))
	T_transfer.append((t1-t0))
	#print x.shape, x_np.shape
	x = reorder_vector(x, rcm_perm, True)
	#import IPython; IPython.embed()
	print 'Relative Error', np.sum((np.asarray(np.dot(AtA, x))[0]-Atb_)**2)/np.sum(Atb**2)
	print "compute+transfer per sample: ", np.array(T_compute)/batch
