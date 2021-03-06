#ifndef _TINYML_OPERATIONS_MATRIXOPCPU_H
#define _TINYML_OPERATIONS_MATRIXOPCPU_H

#include<tuple>
#include<math.h>

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
  void gemmCPU(float *alpha, float *beta, float *omega, float *gamma,
      size_t gamma_row, size_t dim_num, size_t gamma_col,
      bool t_alpha = false,  bool t_beta = false,
      float a = 1, float b = 1);


  /* Reduce matrix to vector
   *
   *  Specification:
   *   - m (row, col)
   *   - v (row) if reduced_dim is 2, else (col)
   */
  void reduceMatrixCPU(float* m, float* v, int row, int col, int reduced_dim);

  /* Subtract two matrixes
   *  omega = alpha - beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   *   - omegarow, col)
   */
  void eleSubtractCPU(float* alpha, float* beta,
      float* omega, int row, int col);

  /* Element Square matrixes
   *  beta = alpha^2
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void eleSquareCPU(float* alpha, float* beta, int row, int col);

  /* reduceToValue
   *  v = sum(alpha)
   *
   *  Specification:
   *   - alpha (row, col)
   */
  void reduceToValueCPU(float* alpha, float* value, int row, int col);

  /* linearOp
   *  beta = alpha * b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOpCPU(float* alpha, float* beta, int row, int col, float scale, float bias = 0);

  /* multiEle 
   *  omega = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   *   - omega (row, col)
   */
  void multiEleCPU(float* alpha, float* beta, float* omega, int row, int col);

  /* multiEleInplace
   *  beta = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   */
  void multiEleInplaceCPU(float* alpha, float* beta, int row, int col);

  /* sigmoidOp 
   *  beta = 1 / (1 + exp(-alpha))
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void sigmoidOpCPU(float* alpha, float* beta, int row, int col);

  /* linearOpInplace 
   *  beta = beta * a + alpha*b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOpInplaceCPU(float* alpha, float* beta, int row, int col, float a, float b, float c = 0);

  /* UpdateWeightWithReg
   *  w = w - delta*lr - reg*w
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void UpdateWeightWithRegCPU(float* delta, float* weight, int row, int col, float lr, float reg);

  /* getCorrectlyRecognized 
   *  get total number of correctly recognized value
   *
   *  Specification:
   *   - predicts (row, col)
   *   - targets (row, col)
   */
  int getCorrectlyRecognizedCPU(float* predicts, float* targets, int row, int col);

  /* softmax
   *  Apply softmax operation
   *   beta[i, :] = softmax(alph[i, :])
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void softmaxCPU(float* alpha, float* beta, int row, int col);

  /* negLogLikelihood
   *  Apply negLogLikelihood operation
   *   loss[i] = negloglikeli(predict[i, :], taget[i, :])
   *   return sum loss[i]
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  float negLogLikelihoodCPU(float* predicts, float* targets, int row, int col);

  void normalizeCPU(float* data, int row, int col, float mean, float std);


  std::tuple<float, float> getStdAndMeanCPU(float *data, shape sh);
}
}

#endif
