import numpy as np
import gpusolver
from scipy.sparse import csr_matrix


def test_all(A, Axbonly=False):
       b = np.ones(A.shape[0], dtype=np.float32)
       #initialize with shape of array
       solver = gpusolver.DnSolver(*A.shape)

       #get numpy result:
       x_np = np.dot(np.linalg.pinv(np.dot(A.T,A)), np.dot(A.T,b))

       #pass flattened array to dense initializer 
       solver.from_dense(A.flatten(order='F'), b)

       #Solve AtAx=Atb using QR (0:QR, 1:cholesky, 2:LU, 3:SVD)
       for i in xrange(3):
              solver.solve(i)
              #retrieve result to host
              x = solver.retrieve()
              print 'Relative Error', np.sum((x-x_np)**2)/np.sum(x_np**2)

       if A.shape[0] == A.shape[1]:
              #Solve Ax=b using QR
              solver.solve_Axb(0)
              x = solver.retrieve()
              print 'Relative Error', np.sum((x-x_np)**2)/np.sum(x_np**2)

       #testing constructor from csr
       Acsr = csr_matrix(A)
       solver.from_csr(Acsr.indptr, Acsr.indices, Acsr.data, b)
       solver.solve(0)

       #retrieve result to host
       x = solver.retrieve()
       print 'Relative Error', np.sum((x-x_np)**2)/np.sum(x_np**2)

def test_Axb(A):
       assert(A.shape[0] == A.shape[1])
       b = np.ones(A.shape[0], dtype=np.float32)
       #initialize with shape of array
       solver = gpusolver.DnSolver(*A.shape)

       #Solve AtAx=Atb using QR (0:QR, 1:cholesky, 2:LU, 3:SVD)
       x_np = np.dot(np.linalg.pinv(np.dot(A.T,A)), np.dot(A.T,A))

       #pass flattened array to dense initializer 
       solver.from_dense(A.flatten(order='F'), b)

       #Solve Ax=b using QR
       solver.solve_Axb(0)
       x = solver.retrieve()
       print 'Relative Error', np.sum((x-x_np)**2)/np.sum(x_np**2)

       #Solve Ax=b using QR
       Acsr = csr_matrix(A)
       solver.from_csr(Acsr.indptr, Acsr.indices, Acsr.data, b)
       solver.solve_Axb(0)
       x = solver.retrieve()
       print 'Relative Error', np.sum((x-x_np)**2)/np.sum(x_np**2)


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
test_Axb(m2)


print "~~~non-square array without solution"
m2 = np.array([[ 5.,  0.,  1.],
       [ 6., -1.,  4.],
       [ 3.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0., -2.,  0.],
       [ 0.,  6.,  0.],
       [ 5., -1.,  7.]]).astype(np.float32)
test_all(m2)