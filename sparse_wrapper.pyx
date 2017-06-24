import numpy as np
cimport numpy as np

#assert sizeof(int) == sizeof(np.int32_t)
#assert sizeof(np.float32) == sizeof(np.float32_t)


cdef extern from "Sparse_manager.hh":
    cdef cppclass C_SpSolver "SpSolver":
        C_SpSolver(np.int32_t, np.int32_t, np.int32_t)
        void prepare_workspace(np.int32_t*, np.int32_t*)
        void from_csr(np.float32_t*, np.float32_t*)
        void solve_Axb_and_retrieve(np.float32_t*)

cdef class SpSolver:
    cdef C_SpSolver* g
    cdef int rows
    cdef int cols
    cdef int batchsize

    def __cinit__(self,  np.int32_t rows, np.int32_t cols, np.int32_t batchsize):
        self.rows, self.cols = rows, cols
        self.batchsize = batchsize
        self.g = new C_SpSolver( self.rows, self.cols, self.batchsize)

    def prepare_workspace(self, np.ndarray[ndim=1, dtype=np.int32_t] indptr, np.ndarray[ndim=1, dtype=np.int32_t] indices):
        self.g.prepare_workspace(&indptr[0], &indices[0])

    def from_csr(self, np.ndarray[ndim=1,dtype=np.float32_t] data, np.ndarray[ndim=1,dtype=np.float32_t] rhs):
        self.g.from_csr(&data[0], &rhs[0])

    def solve_Axb_and_retrieve(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] x = np.zeros(self.cols*self.batchsize, dtype=np.float32)

        self.g.solve_Axb_and_retrieve(&x[0])

        return x
