#include "Solver_manager.hh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cmath> 
#include <cuda_runtime.h>
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include <chrono>
#include <iostream>
#include <complex>
#include <cuComplex.h>
#include "corrMatrix.cu"



DnSolver::DnSolver (int rows_) 
{
    DnSolver::~DnSolver();

    rowsA = rows_;
    colsA = rows_;
    lda = rows_;

    checkCudaErrors(cusolverDnCreate(&cusolverH));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cusolverDnSetStream(cusolverH, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    h_A = (std::complex<float>*)malloc(sizeof(std::complex<float>)*lda*colsA);
    h_V = (std::complex<float>*)malloc(sizeof(std::complex<float>)*rowsA);
    h_S = (float*)malloc(sizeof(float)*colsA);

    checkCudaErrors(cudaMalloc((void **)&d_S, sizeof(float)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_V, sizeof(cuComplex)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(cuComplex)*lda*colsA)); 

    checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int))); 

    checkCudaErrors(cusolverDnCheevd_bufferSize( cusolverH, jobz, uplo, lda, d_A, lda, d_S, &lwork));

    checkCudaErrors(cudaMalloc((void**)&d_work, sizeof(cuComplex)*lwork));


}



void DnSolver::corr_from_vec(std::complex<float>* vec_host_){

    h_V = vec_host_;
    checkCudaErrors(cudaMemcpy(d_V, h_V, sizeof(cuComplex)*lda, cudaMemcpyHostToDevice));
    get_correlation(d_A, d_V, lda, 256);
    //checkMatrix(lda, 1, d_V, lda, "dV");
    //checkMatrix(lda, lda, d_A, lda, "dA");

}



void DnSolver::solve() {
    
    int n = lda;
    int h_info = 0;
    float start, stop;
    float time_solve;

    start = second();

    checkCudaErrors(cusolverDnCheevd( cusolverH, jobz, uplo, n, d_A, lda, d_S, d_work, lwork, devInfo));
    //checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: EVD failed, check %d parameter\n", h_info);
    }

   
    stop = second();
    time_solve = stop - start; 
    //fprintf (stdout, "timing: EVD = %10.6f sec\n", time_solve);

    //if (d_A ) cudaFree(d_A); 
    //if (d_S ) cudaFree(d_S); 
    //if (devInfo) cudaFree(devInfo); 
    //if (d_work ) cudaFree(d_work); 
    //if (cublasHandle ) cublasDestroy(cublasHandle); 
    //if (cusolverH) cusolverDnDestroy(cusolverH); 
}


void DnSolver::retrieve_to(float* h_S)
{
    checkCudaErrors(cudaMemcpy(h_S, d_S, sizeof(float)*colsA, cudaMemcpyDeviceToHost));
}

DnSolver::~DnSolver()
{
    if (cusolverH) { checkCudaErrors(cusolverDnDestroy(cusolverH)); }
    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

    if (h_A) { free(h_A); }
    if (h_S) { free(h_S); }
    if (h_V) { free(h_V); }


    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_S) { checkCudaErrors(cudaFree(d_S)); }
    if (d_V) { checkCudaErrors(cudaFree(d_V)); }

}
