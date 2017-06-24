//#include "cuSolverDn_AtA.cu"
#include "Solver_manager.hh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cmath> 
#include <cuda_runtime.h>
#include "SI.h"
#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"

int linearSolverCHOL(
    cusolverDnHandle_t handle,
    int n,
    const float *Acopy,
    int lda,
    const float *b,
    float *x)
{
    int bufferSize = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    int h_info = 0;
    float start, stop;
    float time_solve;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(cusolverDnSpotrf_bufferSize(handle, uplo, n, (float*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));


    // prepare a copy of A because potrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnSpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: Cholesky factorization failed, check %d parameter\n", h_info);
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusolverDnSpotrs(handle, uplo, n, 1, A, lda, x, n, info));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: cholesky = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info)); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }

    return 0;
}


/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
int linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const float *Acopy,
    int lda,
    const float *b,
    float *x)
{
    int bufferSize = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    float start, stop;
    float time_solve;

    checkCudaErrors(cusolverDnSgetrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed, check %d parameter\n", h_info);
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv));}

    return 0;
}


int linearSolverSVD(
    cusolverDnHandle_t handle, 
    int n,
    const float *Acopy,
    int lda,
    const float *bcopy,
    float *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int m = lda;
    int bufferSize = 0;
    int *info = NULL;
    int h_info = 0;
    float start, stop;
    float time_solve;
    const float one = 1.0;

    // float U[lda*m]; // m-by-m unitary matrix 
    // float VT[lda*n]; // n-by-n unitary matrix
    // float S[n]; //singular value 
    float *d_A = NULL; float *d_SI = NULL; 
    float *d_b = NULL; float *d_S = NULL; 
    float *d_U = NULL; float *d_VT = NULL; 
    float *d_work = NULL; 
    float *d_rwork = NULL; 
    float *d_W = NULL; 
    signed char jobu = 'A'; // all m columns of U 
    signed char jobvt = 'A'; // all n columns of VT 
    // step 1: create cusolverDn/cublas handle 
    checkCudaErrors(cublasCreate(&cublasHandle)); 

    checkCudaErrors(cudaMalloc((void**)&d_A , sizeof(float)*lda*n)); \
    checkCudaErrors(cudaMalloc((void**)&d_b , sizeof(float)*m)); 
    checkCudaErrors(cudaMalloc((void**)&d_S , sizeof(float)*n)); 
    checkCudaErrors(cudaMalloc((void**)&d_SI , sizeof(float)*lda*n)); 
    checkCudaErrors(cudaMalloc((void**)&d_U , sizeof(float)*lda*m)); 
    checkCudaErrors(cudaMalloc((void**)&d_VT , sizeof(float)*lda*n)); 
    checkCudaErrors(cudaMalloc((void**)&info, sizeof(int))); 
    checkCudaErrors(cudaMalloc((void**)&d_W , sizeof(float)*lda*n));
    checkCudaErrors(cudaMemcpy(d_A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice)); //gesvd destroys d_A on exit
    checkCudaErrors(cudaMemcpy(d_b, bcopy, sizeof(float)*m, cudaMemcpyDeviceToDevice));
    
    // checkMatrix(m, n, d_SI, lda, "zero_SI");
    // checkMatrix(m, n , d_A, lda, "SVD_AtA");
    // checkArray(d_b, m, "SVD_Atb");
    checkCudaErrors(cusolverDnSgesvd_bufferSize( handle, m, n, &bufferSize ));
    checkCudaErrors(cudaMalloc((void**)&d_work , sizeof(float)*bufferSize));

    start = second();

    checkCudaErrors(cusolverDnSgesvd( 
        handle, jobu, jobvt, m, n, d_A, lda, d_S, d_U, lda, d_VT, lda, d_work, bufferSize, d_rwork, info));
    //checkCudaErrors(cudaDeviceSynchronize());
    
    // checkCudaErrors(cudaMemcpy(U , d_U , sizeof(float)*lda*m, cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(VT, d_VT, sizeof(float)*lda*n, cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(S , d_S , sizeof(float)*n , cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: SVD failed, check %d parameter\n", h_info);
    }

    // int BLOCK_DIM_X = 32; int BLOCK_DIM_Y = 32;
    // dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
    // dim3 gridDim((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (m + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    // initSIGPU<<<gridDim, blockDim>>>(d_SI, d_S, m, n);
    float epsilon = 1.e-9;
    printf("epsilon = %f \n", epsilon);
    initSI<float>(d_SI, d_S, m, n, epsilon, 256);
    //int initStat = initSICPU(d_SI, d_S, m, n, epsilon);
    // U*S*V*x=b; x = VT*Si*UT*b
    // checkMatrix(m, n, d_SI, lda, "SVD_SI");
    // checkArray(d_S, n, "dS");
    // checkMatrix(m, m, d_U, lda, "SVD_U");
    // checkMatrix(n, n, d_VT, lda, "SVD_VT");
    float al = 1.0;// al =1
    float bet = 0.0;// bet =0
    // checkArray(d_b, n, "db");
    checkCudaErrors(cublasSgemv(cublasHandle,CUBLAS_OP_T, m, m, &al,d_U, m, d_b,1,&bet,d_b,1));
    // checkArray(d_b, n, "dUtb");
    checkCudaErrors(cublasSgemv(cublasHandle,CUBLAS_OP_N, m, n, &al,d_SI, m, d_b,1,&bet,d_b,1));
    // checkArray(d_b, n, "dSiUtb");
    checkCudaErrors(cublasSgemv(cublasHandle,CUBLAS_OP_T, n, n, &al,d_VT, n, d_b, 1,&bet,x,1));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_solve = stop - start; 
    fprintf (stdout, "timing: SVD = %10.6f sec\n", time_solve);
    // checkArray(x, 20, "d_x");

    if (d_A ) cudaFree(d_A); 
    if (d_S ) cudaFree(d_S); 
    if (d_SI ) cudaFree(d_SI);
    if (d_U ) cudaFree(d_U); 
    if (d_VT ) cudaFree(d_VT); 
    if (info) cudaFree(info); 
    if (d_work ) cudaFree(d_work); 
    if (d_rwork) cudaFree(d_rwork); 
    if (d_W ) cudaFree(d_W); 
    if (cublasHandle ) cublasDestroy(cublasHandle); 
    // if (cusolverH) cusolverDnDestroy(cusolverH); 
    return 0;

}

/*
 *  solve A*x = b by QR
 *
 */
int linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const float *Acopy,
    int lda,
    const float *b,
    float *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    float *tau = NULL;
    int h_info = 0;
    float start, stop;
    float time_solve;
    const float one = 1.0;

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnSgeqrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnSormqr_bufferSize(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        NULL,
        x,
        n,
        &bufferSize_ormqr));

    //printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);
    
    bufferSize = (bufferSize_geqrf > bufferSize_ormqr)? bufferSize_geqrf : bufferSize_ormqr ; 

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(float)*n));

// prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

// compute QR factorization
    checkCudaErrors(cusolverDnSgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: QR factorization failed, check %d parameter\n", h_info);
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnSormqr(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        tau,
        x,
        n,
        buffer,
        bufferSize,
        info));

    // x = R \ Q^T*b
    checkCudaErrors(cublasStrsm(
         cublasHandle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         n,
         1,
         &one,
         A,
         lda,
         x,
         n));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: QR = %10.6f sec\n", time_solve);

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (tau   ) { checkCudaErrors(cudaFree(tau)); }

    return 0;
}


DnSolver::DnSolver (int rows_, int cols_) 
{
    DnSolver::~DnSolver();

    rowsA = rows_;
    colsA = cols_;
    lda = rows_;

    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cusparseCreate(&cusparseHandle));
    checkCudaErrors(cusparseCreateMatDescr(&descrA));

    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));
    checkCudaErrors(cusparseSetStream(cusparseHandle, stream));


    h_A = (float*)malloc(sizeof(float)*lda*colsA);
    h_x = (float*)malloc(sizeof(float)*colsA);
    h_b = (float*)malloc(sizeof(float)*rowsA);

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(float)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(float)*lda*colsA)); 


}

void DnSolver::from_dense(float* array_host_, float* rhs_){

    h_A = array_host_;
    h_b = rhs_;
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(float)*lda*colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(float)*rowsA, cudaMemcpyHostToDevice));
}


void DnSolver::from_csr(int* indptr_, int* indices_, float* data_, float* rhs_){
    
    h_b = rhs_;
    h_csrRowPtrA = indptr_;
    h_csrColIndA = indices_;
    h_csrValA = data_;
    baseA = h_csrRowPtrA[0];
    nnzA = h_csrRowPtrA[rowsA] - baseA;
    //checkMatrix(nnzA, 1, h_csrValA, nnzA, "h_valA");
    cusparseStatus_t cpstat;

    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL); 
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    if (d_csrRowPtrA == NULL ){
        printf("allocating pointers \n");
        checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
        checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
        checkCudaErrors(cudaMalloc((void **)&d_csrValA, sizeof(float)*nnzA));
    }
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValA, sizeof(float)*nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(float)*rowsA, cudaMemcpyHostToDevice));

    cpstat = cusparseScsr2dense(
                    cusparseHandle, 
                    rowsA, colsA, 
                    descrA, 
                    d_csrValA, 
                    d_csrRowPtrA,
                    d_csrColIndA, 
                    d_A, rowsA);
    if (cpstat != CUSPARSE_STATUS_SUCCESS) { 
        printf ("%s\n", "CuSparse CSR to dense conversion failed"); 
        return; 
    } 

    //if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
    //if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
    //if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }

    //cpstat = cusparseScsrmm(cusparseHandle,
    //                        CUSPARSE_OPERATION_TRANSPOSE,
    //                        colsA,colsA,rowsA,nnzA, &al,
    //                        descrA, 
    //                        d_csrValA,
    //                        h_csrRowPtrA, 
    //                        d_csrColIndA, rowsA,d_A,rowsA,&bet,dAtA,colsA);


}

void DnSolver::solve(int Func) {
    //printf("step 6: compute AtA \n");
    cublasStatus_t cbstat;
    float* dAtA;
    checkCudaErrors(cudaMalloc(&dAtA, sizeof(float)*colsA*colsA));

    cbstat = cublasSgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,colsA,colsA,rowsA,&al,d_A,rowsA,d_A,rowsA,&bet,dAtA,colsA);

    // printf("step 7: compute At*b \n");
    float* d_Atb;
    checkCudaErrors(cudaMalloc((void **)&d_Atb, sizeof(float)*colsA));
    cbstat = cublasSgemv(cublasHandle,CUBLAS_OP_T,rowsA,colsA,&al,d_A,rowsA,d_b,1,&bet,d_Atb,1);

    //print out for debug
    //checkMatrix(rowsA, colsA , d_A, lda, "A");
    //checkMatrix(rowsA, 1 , d_b, rowsA, "b");
    //checkMatrix(colsA, colsA , dAtA, lda, "AtA");
    //checkMatrix(colsA, 1 , d_Atb, lda, "Atb");

    //if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    //checkCudaErrors(cublasCreate(&cublasHandle));
    //checkCudaErrors(cublasSetStream(cublasHandle, stream));
    //printf("step 8: solves AtA*x = At*b \n");

    if ( 0 == Func )
    {
        linearSolverQR(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else if ( 1 == Func )
    {
        linearSolverCHOL(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else if ( 2 == Func )
    {
        linearSolverLU(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else if ( 3 == Func )
    {
        linearSolverSVD(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else
    {
        fprintf(stderr, "Error: %d is unknown function\n", Func);
        exit(EXIT_FAILURE);
    }
    if (dAtA) { checkCudaErrors(cudaFree(dAtA)); }
    if (d_Atb) { checkCudaErrors(cudaFree(d_Atb)); }
    //if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    //if (d_b) { checkCudaErrors(cudaFree(d_b)); }

}
void DnSolver::solve_Axb(int Func) {


    if ( 0 == Func )
    {
        linearSolverQR(handle, colsA, d_A, colsA, d_b, d_x);
    }
    else if ( 1 == Func )
    {
        linearSolverCHOL(handle, colsA, d_A, colsA, d_b, d_x);
    }
    else if ( 2 == Func )
    {
        linearSolverLU(handle, colsA, d_A, colsA, d_b, d_x);
    }
    else if ( 3 == Func )
    {
        linearSolverSVD(handle, colsA, d_A, colsA, d_b, d_x);
    }
    else
    {
        fprintf(stderr, "Error: %d is unknown function\n", Func);
        exit(EXIT_FAILURE);
    }
    //if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    //if (d_b) { checkCudaErrors(cudaFree(d_b)); }

}
void DnSolver::retrieve_to(float* h_x)
{
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(float)*colsA, cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaFree(d_x));
    //if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    //if (d_b) { checkCudaErrors(cudaFree(d_b)); }
}

DnSolver::~DnSolver()
{
    if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (cusparseHandle) { checkCudaErrors(cusparseDestroy(cusparseHandle)); }
    if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

    if (h_A) { free(h_A); }
    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }

    if (h_csrValA   ) { free(h_csrValA); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }

    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }


    if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
    if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
    if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }
}
