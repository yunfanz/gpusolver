#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include <cuComplex.h>
#include <complex>
class DnSolver{


    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;


    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A

    int lda   = 0; // leading dimension in dense matrix


    std::complex<float> *h_A = NULL; // correlation matrix
    std::complex<float> *h_V = NULL; // correlation vector
    float *h_S = NULL;//eigevalues

    cuComplex *d_A = NULL; // correlation matrix
    cuComplex *d_V = NULL; // correlation vector
    float *d_S = NULL; //eigevalues


    // the constants are used in residual evaluation, r = b - A*x
    const float minus_one = -1.0;
    const float one = 1.0;

    const float al = 1.0;// al =1
    const float bet = 0.0;// bet =0

    int errors = 0;
    int *devInfo = NULL; 
    cuComplex *d_work = NULL; 
    int lwork = 0; 

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER; 


public:


  DnSolver (int rows_) ; // constructor

  ~DnSolver(); // destructor

  void corr_from_vec(std::complex<float>* vec_host_);

  void solve(); 
  //gets results back from the gpu, putting them in the supplied memory location
  void retrieve_to (float* h_S);


};