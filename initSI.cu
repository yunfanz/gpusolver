#include <stdio.h>

#include "SI.h"
#include <cuda_runtime.h>
#include <cmath> 
#include <helper_cuda.h>
template <typename T_ELEM>
__global__ void 
initSIGPU(T_ELEM *SI, T_ELEM *SDiag, int numR, int maxInd, T_ELEM epsilon) {

    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    if (ind < maxInd)
    {
        T_ELEM Sreg = ((SDiag[ind])>epsilon) ? 1./(SDiag[ind]) : 0.0;
        SI[ind+numR*ind] = Sreg; 
    }
}

template <typename T_ELEM>
int initSICPU(T_ELEM *SI, T_ELEM *SDiag, int numR, int numC, T_ELEM epsilon)
{
    T_ELEM *h_SI = NULL; T_ELEM *h_S = NULL;
    T_ELEM val;
    h_SI = (T_ELEM*)malloc(sizeof(T_ELEM)*numR*numC);
    h_S = (T_ELEM*)malloc(sizeof(T_ELEM)*numC);

    checkCudaErrors(cudaMemcpy(h_S, SDiag, sizeof(T_ELEM)*numC, cudaMemcpyDeviceToHost));
    //printArray(h_S, numC);
    memset(h_SI, 0, sizeof(T_ELEM)*numC*numR);

    int maxInd = (numC<numR) ? numC : numR;
    for(int ind = 0 ; ind < maxInd ; ind++)
    {
        val = ((h_S[ind])>epsilon) ? 1./(h_S[ind]) : 0.0;
        h_SI[ind + ind*numR] = val;
    }
    //printMatrix(numR, numC, h_SI, numR, "h_SI");
    checkCudaErrors(cudaMemcpy(SI, h_SI, sizeof(T_ELEM)*numR*numC, cudaMemcpyHostToDevice));

    if (h_S) { free(h_S); }
    if (h_SI) { free(h_SI); }

    return 0;
}
template <typename T_ELEM>
void initSI(T_ELEM *SI, T_ELEM *SDiag, int numR, int numC, T_ELEM epsilon, int threadsPerBlock)
{
    cudaError_t err = cudaSuccess;
    int maxInd = (numC<numR) ? numC : numR;
    checkCudaErrors(cudaMemset(SI, 0.0, sizeof(T_ELEM)*numR*numC));
    int blocksPerGrid =(maxInd + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    initSIGPU<T_ELEM><<<blocksPerGrid, threadsPerBlock>>>(SI, SDiag, numR, maxInd, epsilon);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SI kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template void initSI<double>(double *SI, double *SDiag, int numR, int numC, double epsilon, int threadsPerBlock);
template void initSI<float>(float *SI, float *SDiag, int numR, int numC, float epsilon, int threadsPerBlock);
template __global__ void initSIGPU<float>(float *SI, float *SDiag, int numR, int maxInd, float epsilon);
template __global__ void initSIGPU<double>(double *SI, double *SDiag, int numR, int maxInd, double epsilon);