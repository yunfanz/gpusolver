import numpy as np
cimport numpy as np

#assert sizeof(int) == sizeof(np.int32_t)
#assert sizeof(np.float32) == sizeof(np.float32_t)

cdef extern from "Solver_manager.hh":
    cdef cppclass C_DnSolver "DnSolver":
        C_DnSolver(np.int32_t, np.int32_t)
        void from_dense(np.float32_t*, np.float32_t*)
        void from_csr(np.int32_t*, np.int32_t*, np.float32_t*, np.float32_t*)
        void solve(np.int32_t)
        void solve_Axb(np.int32_t)
        void retrieve_to(np.float32_t*)

cdef class DnSolver:
    cdef C_DnSolver* g
    cdef int rows
    cdef int cols

    def __cinit__(self,  np.int32_t rows, np.int32_t cols):
        self.rows, self.cols = rows, cols
        self.g = new C_DnSolver( self.rows, self.cols)

    def from_dense(self, np.ndarray[ndim=1, dtype=np.float32_t] arr, np.ndarray[ndim=1,dtype=np.float32_t] rhs):
        self.g.from_dense(&arr[0], &rhs[0])

    def from_csr(self, np.ndarray[ndim=1, dtype=np.int32_t] indptr, np.ndarray[ndim=1, dtype=np.int32_t] indices, np.ndarray[ndim=1,dtype=np.float32_t] data, np.ndarray[ndim=1,dtype=np.float32_t] rhs):
        self.g.from_csr(&indptr[0], &indices[0], &data[0], &rhs[0])

    def solve(self, np.int32_t func):
        self.g.solve(func)

    def solve_Axb(self, np.int32_t func):
        self.g.solve_Axb(func)

    def retrieve(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] x = np.zeros(self.cols, dtype=np.float32)

        self.g.retrieve_to(&x[0])

        return x