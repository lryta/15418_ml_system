#ifndef _TINYML_OPERATIONS_MATRIXOP_H
#define _TINYML_OPERATIONS_MATRIXOP_H

#include<tuple>
#include<math.h>

#include"env.h"

#ifdef COMPILE_ISPC
#include "operations/matrixOpISPC.h"
#endif // COMPILE_ISPC

#include "operations/matrixOpCPU.h"

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
      float a = 1, float b = 1);

  /* Reduce matrix to vector
   *
   *  Specification:
   *   - m (row, col)
   *   - v (row) if reduced_dim is 2, else (col)
   */
  void reduceMatrix(float* m, float* v, int row, int col, int reduced_dim);

  /* Subtract two matrixes
   *  omega = alpha - beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   *   - omegarow, col)
   */
  void eleSubtract(float* alpha, float* beta,
      float* omega, int row, int col);

  /* Element Square matrixes
   *  beta = alpha^2
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void eleSquare(float* alpha, float* beta, int row, int col);

  /* reduceToValue
   *  v = sum(alpha)
   *
   *  Specification:
   *   - alpha (row, col)
   */
  void reduceToValue(float* alpha, float* value, int row, int col);

  /* linearOp
   *  beta = alpha * b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOp(float* alpha, float* beta, int row, int col, float scale, float bias = 0);

  /* multiEle 
   *  omega = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   *   - omega (row, col)
   */
  void multiEle(float* alpha, float* beta, float* omega, int row, int col);

  /* multiEleInplace
   *  beta = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   */
  void multiEleInplace(float* alpha, float* beta, int row, int col);

  /* sigmoidOp 
   *  beta = 1 / (1 + exp(-alpha))
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void sigmoidOp(float* alpha, float* beta, int row, int col);

  /* linearOpInplace 
   *  beta = beta * a + alpha*b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOpInplace(float* alpha, float* beta, int row, int col, float a, float b, float c = 0);

  /* UpdateWeightWithReg
   *  w = w - delta*lr - reg*w
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void UpdateWeightWithReg(float* delta, float* weight, int row, int col, float lr, float reg);

  /* getCorrectlyRecognized 
   *  get total number of correctly recognized value
   *
   *  Specification:
   *   - predicts (row, col)
   *   - targets (row, col)
   */
  int getCorrectlyRecognized(float* predicts, float* targets, int row, int col);

  /* softmax
   *  Apply softmax operation
   *   beta[i, :] = softmax(alph[i, :])
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void softmax(float* alpha, float* beta, int row, int col);

  /* negLogLikelihood
   *  Apply negLogLikelihood operation
   *   loss[i] = negloglikeli(predict[i, :], taget[i, :])
   *   return sum loss[i]
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  float negLogLikelihood(float* predicts, float* targets, int row, int col);

  void normalize(float* data, int row, int col, float mean, float std);


  std::tuple<float, float> getStdAndMean(float *data, shape sh);
}
}

#endif
