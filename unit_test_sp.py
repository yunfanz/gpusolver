import numpy as np
import gpusolverSp as gpusolver
from scipy.sparse import csr_matrix


def test_all(A):
       batch = 1000
       b = np.ones(A.shape[0], dtype=np.float32)
       #initialize with shape of array
       AtA = np.dot(A.T,A)
       Atb = np.dot(A.T, b)
       solver = gpusolver.SpSolver(AtA.shape[0],AtA.shape[1], batch)

       #get numpy result:
       x_np = np.dot(np.linalg.pinv(np.dot(A.T,A)), np.dot(A.T,b))

       #testing constructor from csr
       Acsr = csr_matrix(AtA)
       
       solver.prepare_workspace(Acsr.indptr, Acsr.indices)

       dataA = np.hstack([Acsr.data for i in xrange(batch)])
       datab = np.hstack([b for i in xrange(batch)])
       solver.from_csr(dataA, datab)

       #retrieve result to host
       
       print solver.solve_Axb_and_retrieve()
       #print x.shape
       #print 'Relative Error', np.sum((x-x_np)**2)/np.sum(x_np**2)



print "#################### gpusolver unit-test ##########################"
print "~~~Testing all functionalities with square array"
m1 = np.array([[ 5.,  0.],
       [ 6., -1.]], dtype=np.float32)
test_all(m1)

print "~~~non-square array with solution"
m2 = np.array([[ 1.,  1.,  0.],
       [ 0.,  1.,  1.],
       [ 1.,  0.,  1.],
       [ 0.,  2.,  0.]], dtype=np.float32)

test_all(m2)

print "~~~non-square array with solution, testing only solving"
m1 = m2
m2 = np.dot(m1.T,m1)
b2 = np.dot(m1.T,np.ones(m1.shape[0], dtype=np.float32))
test_all(m2)


print "~~~non-square array without solution"
m2 = np.array([[ 5.,  0.,  1.],
       [ 6., -1.,  4.],
       [ 3.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0., -2.,  0.],
       [ 0.,  6.,  0.],
       [ 5., -1.,  7.]]).astype(np.float32)
test_all(m2)