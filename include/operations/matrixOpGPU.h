#ifndef _TINYML_OPERATIONS_MATRIXOPGPU_H
#define _TINYML_OPERATIONS_MATRIXOPGPU_H
namespace TinyML{

namespace matrix{

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
      bool t_alpha = false,  bool t_beta = false,
      float a = 1, float b = 1);

}
}

#endif
