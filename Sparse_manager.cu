#include "Sparse_manager.hh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cmath> 
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include "cusolverSp.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include <chrono>
#include <iostream>

#define imin( x, y ) ((x)<(y))? (x) : (y) 


SpSolver::SpSolver (int rows_, int cols_, int batch_)
{
    rowsA = rows_;
    colsA = cols_;
    batchSize = batch_;

    checkCudaErrors(cusolverSpCreate(&cusolverSpH));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cusparseCreate(&cusparseHandle));
    checkCudaErrors(cusparseCreateMatDescr(&descrA));

    checkCudaErrors(cusolverSpSetStream(cusolverSpH, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));
    checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

    //h_x = (float*)malloc(sizeof(float)*colsA*batchSize); //maybe this is unncessary
    h_b = (float*)malloc(sizeof(float)*rowsA*batchSize);


}

void SpSolver::prepare_workspace(int* indptr_, int* indices_){

    h_csrRowPtrA = indptr_;
    h_csrColIndA = indices_;
    baseA = h_csrRowPtrA[0];
    nnzA = h_csrRowPtrA[rowsA] - baseA;
    if (d_csrRowPtrA == NULL ){
        printf("allocating pointers \n");
        checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
        checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
        checkCudaErrors(cudaMalloc((void**)&d_csrValA , sizeof(float)*nnzA*batchSize));

    }
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA, cudaMemcpyHostToDevice));
    cusparseCreateMatDescr(&descrA); 
    if (baseA) {cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); }
    else{
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    }
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL); // A must be a general matrix
    cusolverSpCreateCsrqrInfo(&info);

    cusolverSpXcsrqrAnalysisBatched( cusolverSpH, 
                                    rowsA, colsA, 
                                    nnzA, descrA, 
                                    d_csrRowPtrA, d_csrColIndA, info);


    //Find "proper" batchSize 
    // get available device memory 
    size_t free_mem = 0; 
    size_t total_mem = 0; 
    checkCudaErrors(cudaMemGetInfo( &free_mem, &total_mem )); 

    batchSizeMax = 2; 
    while(batchSizeMax < batchSize){ 
        printf("batchSizeMax = %d\n", batchSizeMax); 
        cusolverSpScsrqrBufferInfoBatched( cusolverSpH, 
                                            rowsA, colsA, nnzA, descrA, 
                                            d_csrValA, d_csrRowPtrA, d_csrColIndA, 
                                            batchSizeMax, 
                                            info, 
                                            &size_internal, 
                                            &size_qr); 

        if ( (size_internal + size_qr) > free_mem ){  //batchSizeMax exceeds hardware limit, so cut it by half. 
            batchSizeMax /= 2; 
            break; 
        } 
        batchSizeMax *= 2; // float batchSizMax and try it again. 
    } 

    //Try to squeeze out last bit of memory
    if (batchSizeMax > 1){
        batchSizeMax += batchSizeMax/2;
        cusolverSpScsrqrBufferInfoBatched( cusolverSpH, 
                                            rowsA, colsA, nnzA, descrA, 
                                            d_csrValA, d_csrRowPtrA, d_csrColIndA, 
                                            batchSizeMax, 
                                            info, 
                                            &size_internal, 
                                            &size_qr); 

        if ( (size_internal + size_qr) > free_mem ){  //batchSizeMax exceeds hardware limit
            batchSizeMax -= batchSizeMax/3;
        } 

    }

    // correct batchSizeMax such that it is not greater than batchSize. 
    batchSizeMax = imin(batchSizeMax, batchSize); 
    printf("batchSizeMax = %d\n", batchSizeMax);

    // Need to call cusolverDcsrqrBufferInfoBatched again with batchSizeMax 
    // to fix batchSize used in numerical factorization.
    cusolverSpScsrqrBufferInfoBatched( cusolverSpH, 
                                        rowsA, colsA, nnzA, descrA, 
                                        d_csrValA, d_csrRowPtrA, d_csrColIndA, 
                                        batchSizeMax, 
                                        info, 
                                        &size_internal, 
                                        &size_qr); 

    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal); 
    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr); 

    checkCudaErrors(cudaMalloc((void**)&buffer_qr, size_qr));

    if (d_x == NULL)
    {
        checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(float)*colsA*batchSize));
        checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float)*rowsA*batchSize));
    }
}




void SpSolver::from_csr(float* data_, float* rhs_){
    
    h_b = rhs_;
    h_csrValABatch = data_;
}

void SpSolver::solve_Axb_and_retrieve(float* python_x) {


    for(int idx = 0 ; idx < batchSize; idx += batchSizeMax)
    { 
        // current batchSize 'cur_batchSize' is the batchSize used in numerical factorization 
        const int cur_batchSize = imin(batchSizeMax, batchSize - idx); 
        printf("current batchSize = %d\n", cur_batchSize); 

        // copy part of Aj and bj to device 
        checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValABatch + idx*nnzA, sizeof(float) * nnzA * cur_batchSize, cudaMemcpyHostToDevice)); 
        checkCudaErrors(cudaMemcpy(d_b, h_b + idx*rowsA, sizeof(float) * rowsA * cur_batchSize, cudaMemcpyHostToDevice)); 

        //debugging:
        //checkMatrix(nnzA, cur_batchSize, d_csrValA, nnzA, "h_valA");
        //checkMatrix(rowsA, cur_batchSize, d_b, rowsA, "h_b");

        // solve part of Aj*xj = bj 
        cusolverSpScsrqrsvBatched( cusolverSpH, 
                                    rowsA, colsA, nnzA, descrA,
                                    d_csrValA, d_csrRowPtrA, d_csrColIndA, 
                                    d_b, d_x, 
                                    cur_batchSize, 
                                    info, buffer_qr); 
        // copy part of xj back to host 
        //checkMatrix(colsA, cur_batchSize, d_x, colsA, "d_x");
        checkCudaErrors(cudaMemcpy(python_x + idx*colsA, d_x, sizeof(float)*colsA*cur_batchSize, cudaMemcpyDeviceToHost)); 
    }

}

SpSolver::~SpSolver()
{
    if (cusolverSpH) { checkCudaErrors(cusolverSpDestroy(cusolverSpH)); }
    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (cusparseHandle) { checkCudaErrors(cusparseDestroy(cusparseHandle)); }
    if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

    if (h_A) { free(h_A); }
    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }

    if (h_csrValABatch   ) { free(h_csrValABatch); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }

    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }


    if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
    if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
    if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }
}
