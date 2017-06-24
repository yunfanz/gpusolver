#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverSp.h"
#include "helper_cuda.h"


class SpSolver{


    cusolverSpHandle_t cusolverSpH = NULL;
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descrA = 0;
    csrqrInfo_t info = NULL; 
    size_t size_qr = 0; 
    size_t size_internal = 0; 
    void *buffer_qr = NULL; // working space for numerical factorization 



    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format
    int lda   = 0; // leading dimension in dense matrix
    int batchSize = 1;
    int batchSizeMax = 1;

    // CSR(A) 
    int *h_csrRowPtrA = NULL;
    int *h_csrColIndA = NULL;
    float *h_csrValABatch = NULL;
    // // CSC(A) from I/O
    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    float *d_csrValA = NULL;

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
    //float* dAtA = NULL;
    //float* d_Atb = NULL;

    // the constants are used in residual evaluation, r = b - A*x
    const float minus_one = -1.0;
    const float one = 1.0;

    const float al = 1.0;// al =1
    const float bet = 0.0;// bet =0

    // float x_inf = 0.0;
    // float r_inf = 0.0;
    // float A_inf = 0.0;
    // float b_inf = 0.0;
    // float Ax_inf = 0.0;
    // float tr_inf = 0.0;
    int errors = 0;


public:


  SpSolver (int rows_, int cols_, int batch_) ; 

  ~SpSolver(); // destructor

  void prepare_workspace(int* indptr_, int* indices_);

  void from_csr(float* data_, float* rhs_);
  void solve_Axb_and_retrieve(float* h_x); // for square matrices, only use qr or lu if non-symmetric


};