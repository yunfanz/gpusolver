import numpy as np
import gpusolver
from scipy.sparse import csr_matrix


print "#################### gpusolver unit-test ##########################"
print "Testing all functionalities with square array"
m1 = np.array([[ 5.,  0.],
       [ 6., -1.]], dtype=np.float32)
b1 = np.ones(2, dtype=np.float32)
#initialize with shape of array
solver = gpusolver.DnSolver(np.int32(2), np.int32(2))
#pass flattened array to dense initializer 
solver.from_dense(m1.flatten(order='F'), b1)
#Solve AtAx=Atb using QR (0:QR, 1:cholesky, 2:LU, 3:SVD)
solver.solve(0)
#retrieve result to host
x = solver.retrieve()
print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))
#Solve Ax=b using QR
solver.solve_Axb(0)
x = solver.retrieve()
print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))

m1csr = csr_matrix(m1)
#pass sparse matrix in csr format to constructor
solver.from_csr(m1csr.indptr, m1csr.indices, m1csr.data, b1)
for i in xrange(4):
	solver.solve(i)
	x = solver.retrieve()
	print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))


print "Testing all functionalities with non-square array"
m2 = np.array([[ 5.,  0.,  1.],
       [ 6., -1.,  4.],
       [ 3.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0., -2.,  0.],
       [ 0.,  6.,  0.],
       [ 5., -1.,  7.]]).astype(np.float32)
solver = gpusolver.DnSolver(np.int32(m2.shape[0]), np.int32(m2.shape[1]))

m2csr = csr_matrix(m2)
b2 = np.ones(m2.shape[0], dtype=np.float32)
#solver.from_dense(m2.flatten(order='F'), b2)
solver.from_csr(m2csr.indptr, m2csr.indices, m2csr.data, b2)
for i in xrange(4):
	solver.solve(i)
	x = solver.retrieve()
	print x
x_np = np.dot(np.linalg,pinv(np.dot(m2.T,m2)), np.dot(m2.T,b2))
print "x_np:", x_np
