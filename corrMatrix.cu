#define BLOCK_SIZE 16
#define BLOCK_VOLUME 16*16
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#define INDEX(j,i,ld) ((j) * ld + (i))
#define C_INDEX(i,j,ld) ((j) * ld + (i))
#define E (float) (2.7182818284)

 __global__ void compute_M(cuComplex* cM, cuComplex* cV, int w)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; 
  int bx = blockIdx.x;   int by = blockIdx.y; 
  int bdx = blockDim.x;  int bdy = blockDim.y; 
  int i = bdx * bx + tx; int j = bdy * by + ty; 
  int p = INDEX(j,i,w);
  if (j >= w || i >= w ) return;
  if (j > i){
    cM[p] = cV[j-i];
  }
  else {
    cM[p] = cuConjf(cV[i-j]);
  }
}

void get_correlation(cuComplex* cM, cuComplex* cV, int w, int threadsPerBlock)
{
    cudaError_t err = cudaSuccess;
    int blocksPerGrid =(w + threadsPerBlock - 1) / threadsPerBlock;
    int side = sqrt(threadsPerBlock);
    dim3 dimBlock(side, side);
    dim3 dimGrid;
    dimGrid.x = (w + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (w + dimBlock.y - 1) / dimBlock.y;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", dimBlock.x, dimGrid.x);
    compute_M<<<dimGrid, dimBlock>>>(cM, cV, w);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}