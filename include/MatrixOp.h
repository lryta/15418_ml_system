
namespace MLLib{

namespace matrix{


  #define pos(i, j, col_num) (col_num * i + j)

  /* General matrix multiplication
   *  gamma = a * lph * beta + omega * b
   *
   * Dim:
   *  - alph(alpha_row, alpha_col)
   *  - beta(alpha_col, beta_col)
   *  - omega(1, beta_col)
   *  - gamma(alpha_row, beta_col)
   */
  void gemm(float *alph, float *beta, float *omega, float *gamma,
      float a, float b, size_t alpa_row, size_t alpa_col, size_t beta_col) {
  }
}

}
