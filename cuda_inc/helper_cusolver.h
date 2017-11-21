/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef HELPER_CUSOLVER
#define HELPER_CUSOLVER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>
#include <complex>
#include <cuComplex.h>
#include "cusparse.h"

#define SWITCH_CHAR             '-'

void printMatrix(int m, int n,std::complex<float>*A, int lda, char* name) 
{   
    printf("%s \n", name);
    for(int row = 0 ; row < m ; row++)
    {
        for(int col = 0 ; col < n ; col++)
        { 
            std::complex<float> Areg = A[row + col*lda]; 
            printf("%.3f + %.3f j ", real(Areg), imag(Areg)); 
            if (col == n-1) printf("\n") ;
        } 
    } 
} 
void checkMatrix(int m, int n, cuComplex *A, int lda, char* name)
{
    std::complex<float>* hA;
    hA = (std::complex<float>*)malloc(sizeof(std::complex<float>)*m*n);
    checkCudaErrors(cudaMemcpy(hA, A, sizeof(cuComplex)*m*n, cudaMemcpyDeviceToHost));
    printMatrix(m, n, hA, m, name);
    if (hA) free(hA);
}
void printArray(float *ptr, size_t length)            
{         
    //for statment to print values using array             
    size_t i = 0;
    for( ; i < length; ++i )  printf("%f \n", ptr[i]);        
}  
void printArray(double *ptr, size_t length)            
{         
    //for statment to print values using array             
    size_t i = 0;
    for( ; i < length; ++i )  printf("%f \n", ptr[i]);        
}
void checkArray(double *ptr, size_t length, const char* name)
{
    printf("%s \n", name);
    double* hp;
    hp = (double*)malloc(sizeof(double)*length);
    checkCudaErrors(cudaMemcpy(hp, ptr, sizeof(double)*length, cudaMemcpyDeviceToHost));
    printArray(hp, length);
    if (hp) free(hp);
}

struct  testOpts {
    char *sparse_mat_filename;   // by switch -F<filename>
    const char *testFunc; // by switch -R<name>
    const char *reorder; // by switch -P<name>
    int lda; // by switch -lda<int>
};

double vec_norminf(int n, const double *x)
{
    double norminf = 0;
    for(int j = 0 ; j < n ; j++){
        double x_abs = fabs(x[j]);
        norminf = (norminf > x_abs)? norminf : x_abs;
    }
    return norminf;
}

float vec_norminf(int n, const float *x)
{
    float norminf = 0;
    for(int j = 0 ; j < n ; j++){
        float x_abs = fabs(x[j]);
        norminf = (norminf > x_abs)? norminf : x_abs;
    }
    return norminf;
}
/*
 * |A| = max { |A|*ones(m,1) }
 */
double mat_norminf(
    int m,
    int n,
    const double *A,
    int lda)
{
    double norminf = 0;
    for(int i = 0 ; i < m ; i++){
        double sum = 0.0;
        for(int j = 0 ; j < n ; j++){
           double A_abs = fabs(A[i + j*lda]);
           sum += A_abs;
        }
        norminf = (norminf > sum)? norminf : sum;
    }
    return norminf;
}
float mat_norminf(
    int m,
    int n,
    const float *A,
    int lda)
{
    float norminf = 0;
    for(int i = 0 ; i < m ; i++){
        float sum = 0.0;
        for(int j = 0 ; j < n ; j++){
           float A_abs = fabs(A[i + j*lda]);
           sum += A_abs;
        }
        norminf = (norminf > sum)? norminf : sum;
    }
    return norminf;
}
/*
 * |A| = max { |A|*ones(m,1) }
 */
double csr_mat_norminf(
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA)
{
    const int baseA = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(descrA))? 1:0;

    double norminf = 0;
    for(int i = 0 ; i < m ; i++){
        double sum = 0.0;
        const int start = csrRowPtrA[i  ] - baseA;
        const int end   = csrRowPtrA[i+1] - baseA;
        for(int colidx = start ; colidx < end ; colidx++){
            // const int j = csrColIndA[colidx] - baseA; 
           double A_abs = fabs( csrValA[colidx] );
           sum += A_abs;
        }
        norminf = (norminf > sum)? norminf : sum;
    }
    return norminf;
}


void display_matrix(
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA)
{
    const int baseA = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(descrA))? 1:0;

    printf("m = %d, n = %d, nnz = %d, matlab base-1\n", m, n, nnzA);

    for(int row = 0 ; row < m ; row++){
        const int start = csrRowPtrA[row  ] - baseA;
        const int end   = csrRowPtrA[row+1] - baseA;
        for(int colidx = start ; colidx < end ; colidx++){
            const int col = csrColIndA[colidx] - baseA;
            double Areg = csrValA[colidx];
            printf("A(%d, %d) = %20.16E\n", row+1, col+1, Areg);
        }
    }
}


#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second (void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency (&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter (&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() / 1000.0;
    }
}

#elif defined(__linux) || defined(__QNX__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

#elif defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/sysctl.h>
double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif

#endif

