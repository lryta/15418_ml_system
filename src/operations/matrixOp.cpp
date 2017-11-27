#include<math.h>
#include<operations/MatrixOp.h>

namespace TinyML{

namespace matrix{

  #define pos(i, j, col_num) ((col_num) * (i) + (j))

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
        gamma[pos(i, j, gamma_col)] = value * a + (omega!=NULL)?(omega[j] * b):(0);
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

  /* Subtract two matrixes
   *  omega = alpha - beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   *   - omega (row, col)
   */
  void eleSubtract(float* alpha, float* beta,
      float* omega, int row, int col) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        omega[pos(i, j, col)] = alpha[pos(i, j, col)] - beta[pos(i, j, col)];
  }

  /* sigmoidOp 
   *  beta = 1 / (1 + exp(-alpha))
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void sigmoidOp(float* alpha, float* beta, int row, int col) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] = 1 / (1 + exp(-alpha[pos(i, j, col)]));
  }

  /* Element Square matrixes
   *  beta = alpha^2
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void eleSquare(float* alpha, float* beta, int row, int col) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] = pow(alpha[pos(i, j, col)],2);
  }

  /* reduceToValue
   *  v = sum(alpha)
   *
   *  Specification:
   *   - alpha (row, col)
   */
  void reduceToValue(float* alpha, float* value, int row, int col) {
    float tmp = 0;
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        tmp += alpha[pos(i, j, col)];
    *value = tmp;
  }

  /* linearOp
   *  beta = alpha * b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOp(float* alpha, float* beta, int row,
      int col, float scale, float bias = 0) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] = alpha[pos(i, j, col)] * scale + bias;
  }

  /* multiEle 
   *  omega = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   *   - omega (row, col)
   */
  void multiEle(float* alpha, float* beta, float* omega, int row, int col) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        omega[pos(i, j, col)] += alpha[pos(i, j, col)] * beta[pos(i, j, col)];
  }

  /* multiEleInplace
   *  beta = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   */
  void multiEleInplace(float* alpha, float* beta, int row, int col) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] *= alpha[pos(i, j, col)];
  }

  /* linearOpInplace 
   *  alpha = alpha * a + beta*b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOpInplace(float* alpha, float* beta, int row, int col, float a, float b, float c = 0) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] = beta[pos(i, j, col)] * a
          alpha[pos(i, j, col)] * b + c;
  }

  /* getCorrectlyRecognized 
   *  get total number of correctly recognized value
   *
   *  Specification:
   *   - predicts (row, col)
   *   - targets (row, col)
   */
  int getCorrectlyRecognized(float* predicts, float* targets, int row, int col) {
    int matched_num = 0, max_p_idx, max_p_idx;
    float max_p_value,  max_p_value;

    for (int i = 0; i < row; ++i) {
      max_p_value = predicts[pos(i, 0, col)];
      max_p_idx = 0;
      max_t_value = targets[pos(i, 0, col)];
      max_t_idx = 0;

      for (int j = 1; j < col; ++j) {
        if (max_p_value < predicts[pos(i, j, col)]) {
          max_p_value = predicts[pos(i, j, col)];
          max_p_idx = j;
        }

        if (max_t_value < targets[pos(i, j, col)]) {
          max_t_value = targets[pos(i, j, col)];
          max_t_idx = j;
        }
      }
      matched_num += (max_t_idx == max_p_idx);

    }

    return matched_num;
  }

  /* softmax
   *  Apply softmax operation
   *   beta[i, :] = softmax(alph[i, :])
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void softmax(float* alpha, float* beta, int row, int col) {
    for (int i = 0; i < row; ++i) {
      float sum = 0;
      for (int j = 0; j < col; ++j)
        sum += exp(alph[pos(i, j, col)]);
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] = exp(alph[pos(i, j, col)]) / sum;
    }
  }

  /* negLogLikelihood
   *  Apply negLogLikelihood operation
   *   loss[i] = negloglikeli(predict[i, :], taget[i, :])
   *   return sum loss[i]
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  float negLogLikelihood(float* predicts, float* targets, int row, int col) {
    float loss = 0;
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        loss += targets[pos(i, j, col)] * log(alph[pos(i, j, col)]);
    return -loss;
  }

}
}
