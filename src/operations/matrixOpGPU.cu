#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cstdio>

#include "operations/matrixOpGPU.h"

namespace TinyML{

namespace matrix{

#define pos(i, j, col_num) ((col_num) * (i) + (j))

__global__ void gemmKernel(float *alpha, float *beta, float *omega, float *gamma,
    size_t gamma_row, size_t dim_num, size_t gamma_col,
    bool t_alpha = false,  bool t_beta = false,
    float a = 1, float b = 1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  float value = 0;
  for (int k = 0; k < dim_num; ++k) {
    value += alpha[t_alpha?pos(k, i, gamma_row):pos(i, k, dim_num)]
      * beta[t_beta?pos(j, k, dim_num):pos(k, j, gamma_col)];
  }
  gamma[pos(i, j, gamma_col)] = value * a + ((omega!=NULL)?(omega[j] * b):(0));
}

/* General matrix multiplication
 *  gamma = a * alpha * beta + omega * b
 *
 * Specification:
 *  - alpha(gamma_row, dim_num), row & col exchange on t_alpha
 *  - beta(dim_num, gamma_col), row & col exchange on t_alpha
 *  - omega(gamma_col)
 *  - gamma(gamma_row, gamma_col)
 *
 * Description:
 *  - Omega could be NULL
 */
void gemmGPU(float *alpha, float *beta, float *omega, float *gamma,
    size_t gamma_row, size_t dim_num, size_t gamma_col,
    bool t_alpha,  bool t_beta,
    float a, float b) {
  // 256 threads per block is a healthy number
  int threadsPerBlockx = 16;
  int threadsPerBlocky = 2;
  if (gamma_col % 4 == 0)
    threadsPerBlocky = 4;
  if (gamma_col % 16 == 0)
    threadsPerBlocky = 16;
  dim3 threadsPerBlock(threadsPerBlockx, threadsPerBlocky);

  // 256 block
  dim3 blocks(gamma_row/threadsPerBlockx, gamma_col/threadsPerBlocky);

  gemmKernel<<<blocks, threadsPerBlock>>>(alpha, beta, omega, gamma, gamma_row, dim_num, gamma_col, t_alpha, t_beta, a, b);

  cudaDeviceSynchronize();
}

}
}
