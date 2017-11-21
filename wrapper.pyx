import numpy as np
cimport numpy as np

#assert sizeof(int) == sizeof(np.int32_t)
#assert sizeof(np.float32) == sizeof(np.float32_t)


cdef extern from "Solver_manager.hh":
    cdef cppclass C_DnSolver "DnSolver":
        C_DnSolver(np.int32_t)
        void corr_from_vec(np.complex64_t*)
        void solve()
        void retrieve_to(np.float32_t*)

cdef class DnSolver:
    cdef C_DnSolver* g
    cdef int rows
    cdef int cols
    

    def __cinit__(self,  np.int32_t rows):
        self.rows = rows
        self.cols = rows
        self.g = new C_DnSolver( self.rows)

    def corr_from_vec(self, np.ndarray[ndim=1, dtype=np.complex64_t] arr):
        self.g.corr_from_vec(&arr[0])

    def solve(self):
        self.g.solve()

    def retrieve(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] x = np.zeros(self.cols, dtype=np.float32)

        self.g.retrieve_to(&x[0])

        return x
