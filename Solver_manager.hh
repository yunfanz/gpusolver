#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
class DnSolver{


    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;


    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format
    int lda   = 0; // leading dimension in dense matrix

    // CSR(A) 
    int *h_csrRowPtrA = NULL;
    int *h_csrColIndA = NULL;
    float *h_csrValA = NULL;
    // // CSC(A) from I/O
    // // int *h_cscColPtrA = NULL;
    // // int *h_cscRowIndA = NULL;
    // // float *h_cscValA = NULL;

    float *h_A = NULL; // dense matrix from CSR(A)
    float *h_x = NULL; // a copy of d_x
    float *h_b = NULL; // b = ones(m,1)
    // float *h_r = NULL; // r = b - A*x, a copy of d_r
    // float *h_tr = NULL;

    float *d_A = NULL; // a copy of h_A
    float *d_x = NULL; // x = A \ b
    float *d_b = NULL; // a copy of h_b
    // float *d_r = NULL; // r = b - A*x
    // float *d_tr = NULL; // tr = Atb - AtA*x

    // the constants are used in residual evaluation, r = b - A*x
    const float minus_one = -1.0;
    const float one = 1.0;

    // float x_inf = 0.0;
    // float r_inf = 0.0;
    // float A_inf = 0.0;
    // float b_inf = 0.0;
    // float Ax_inf = 0.0;
    // float tr_inf = 0.0;
    int errors = 0;


public:


  DnSolver (int rows_, int cols_) ; // constructor (copies to GPU)

  ~DnSolver(); // destructor

  void from_dense(float* array_host_, float* rhs_);

  void from_csr(int* indptr_, int* indices_, float* data_, float* rhs_);

  void solve(int func); // AtA solver for non-square matrices
  void solve_Axb(int func); // for square matrices, only use qr or lu if non-symmetric

  //gets results back from the gpu, putting them in the supplied memory location
  void retrieve_to (float* h_x);


};