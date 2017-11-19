
namespace MLLib{

namespace matrix{

  #define pos(i, j, col_num) ((col_num) * (i) + (j))

  /* General matrix multiplication
   *  gamma = a * lph * beta + omega * b
   *
   * Specification:
   *  - alpha(gamma_row, dim_num), row & col exchange on t_alpha
   *  - beta(dim_num, gamma_col), row & col exchange on t_alpha
   *  - omega(1, gamma_col)
   *  - gamma(gamma_row, gamma_col)
   *
   * Description:
   *  - Omega could be NULL
   */
  void gemm(float *alpha, float *beta, float *omega, float *gamma,
      size_t gamma_row, size_t dim_num, size_t gamma_col,
      bool t_alpha = false,  bool t_beta = false,
      float a = 1, float b = 1) {
    float value;
    for (size_t i = 0; i < gamma_row; ++i)
      for (size_t j = 0; j < gamma_col; ++j) {
        value = 0;
        for (size_t k = 0; k < dim_num; ++k)
          value += alpha[t_alpha?pos(k, i, gamma_row):pos(i, k, dim_num)]
            * gamma_col[t_beta?pos(j, k, dim_num):pos(k, j, gamma_col)];
        gamma[pos(i, j, gamma_col)] = value * a + (omega!=NULL)?omega[j] * b:(0);
      }
  }


  /* Reduce matrix to vector
   *
   *  Specification:
   *   - m (row, col)
   *   - v (row) if reduced_dim is 2, else (col)
   */
  void reduceMatrix(float* m, float* v, int row, int col, int reduced_dim) {
    assert(reduced_dim <= 2);
    float value = 0;
    if (reduced_dim == 1) {
      for (int i = 0; i < col; ++i) {
        for (int j = 0; j < row; ++j)
          value += m[pos(j, i, col)];
        v[i] = value;
      }
    } else {
      for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j)
          value += m[pos(i, j, col)];
        v[i] = value;
      }
    }
  }

}
}
