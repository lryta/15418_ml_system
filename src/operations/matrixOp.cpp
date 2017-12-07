#include <cassert>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "shape.h"
#include "operations/matrixOp.h"

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
      bool t_alpha,  bool t_beta, float a, float b) {
#ifdef COMPILE_ISPC
    gemmISPC(alpha, beta, omega, gamma, gamma_row,
        dim_num, gamma_col, t_alpha, t_beta, a, b);
#else
    gemmCPU(alpha, beta, omega, gamma, gamma_row,
        dim_num, gamma_col, t_alpha, t_beta, a, b);
#endif
  }


  /* Reduce matrix to vector
   *
   *  Specification:
   *   - m (row, col)
   *   - v (row) if reduced_dim is 2, else (col)
   */
  void reduceMatrix(float* m, float* v, int row, int col, int reduced_dim) {
    assert(reduced_dim <= 2);
#ifdef COMPILE_ISPC
    reduceMatrixISPC(m, v, row, col, reduced_dim);
#else
    reduceMatrixCPU(m, v, row, col, reduced_dim);
#endif
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
#ifdef COMPILE_ISPC
    eleSubtractISPC(alpha, beta, omega, row, col);
#else
    eleSubtractCPU(alpha, beta, omega, row, col);
#endif
  }

  /* sigmoidOp 
   *  beta = 1 / (1 + exp(-alpha))
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void sigmoidOp(float* alpha, float* beta, int row, int col) {
#ifdef COMPILE_ISPC
    sigmoidOpISPC(alpha, beta, row, col);
#else
    sigmoidOpCPU(alpha, beta, row, col);
#endif
  }

  /* Element Square matrixes
   *  beta = alpha^2
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void eleSquare(float* alpha, float* beta, int row, int col) {
#ifdef COMPILE_ISPC
    eleSquareISPC(alpha, beta, row, col);
#else
    eleSquareCPU(alpha, beta, row, col);
#endif
  }

  /* reduceToValue
   *  v = sum(alpha)
   *
   *  Specification:
   *   - alpha (row, col)
   */
  void reduceToValue(float* alpha, float* value, int row, int col) {
#ifdef COMPILE_ISPC
    reduceToValueISPC(alpha, value, row, col);
#else
    reduceToValueCPU(alpha, value, row, col);
#endif
  }

  /* linearOp
   *  beta = alpha * scale + bias
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOp(float* alpha, float* beta, int row,
      int col, float scale, float bias) {
#ifdef COMPILE_ISPC
    linearOpISPC(alpha, beta, row, col, scale, bias);
#else
    linearOpCPU(alpha, beta, row, col, scale, bias);
#endif
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
#ifdef COMPILE_ISPC
    multiEleISPC(alpha, beta, omega, row, col);
#else
    multiEleCPU(alpha, beta, omega, row, col);
#endif
  }

  /* multiEleInplace
   *  beta = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   */
  void multiEleInplace(float* alpha, float* beta, int row, int col) {
#ifdef COMPILE_ISPC
    multiEleInplaceISPC(alpha, beta, row, col);
#else
    multiEleInplaceCPU(alpha, beta, row, col);
#endif
  }

  /* linearOpInplace 
   *  beta = beta * a + alpha*b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOpInplace(float* alpha, float* beta, int row, int col, float a, float b, float c) {
#ifdef COMPILE_ISPC
    linearOpInplaceISPC(alpha, beta, row, col, a, b, c);
#else
    linearOpInplaceCPU(alpha, beta, row, col, a, b, c);
#endif
  }

  /* UpdateWeightWithReg
   *  w = w - delta*lr - reg*w
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void UpdateWeightWithReg(float* delta, float* weight, int row, int col, float lr, float reg) {
#ifdef COMPILE_ISPC
    UpdateWeightWithRegISPC(delta, weight, row, col, lr, reg);
#else
    UpdateWeightWithRegCPU(delta, weight, row, col, lr, reg);
#endif
  }

  /* getCorrectlyRecognized 
   *  get total number of correctly recognized value
   *
   *  Specification:
   *   - predicts (row, col)
   *   - targets (row, col)
   */
  int getCorrectlyRecognized(float* predicts, float* targets, int row, int col) {
#ifdef COMPILE_ISPC
    return getCorrectlyRecognizedISPC(predicts, targets, row, col);
#else
    return getCorrectlyRecognizedCPU(predicts, targets, row, col);
#endif
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
#ifdef COMPILE_ISPC
    softmaxISPC(alpha, beta, row, col);
#else
    softmaxCPU(alpha, beta, row, col);
#endif
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
#ifdef COMPILE_ISPC
    return negLogLikelihoodISPC(predicts, targets, row, col);
#else
    return negLogLikelihoodCPU(predicts, targets, row, col);
#endif
  }

  void normalize(float* data, int row, int col, float mean, float std) {
#ifdef COMPILE_ISPC
    normalizeISPC(data, row, col, mean, std);
#else
    normalizeCPU(data, row, col, mean, std);
#endif
  }

  std::tuple<float, float> getStdAndMean(float *data, shape sh) {
    // No ISPC impl
    return getStdAndMeanCPU(data, sh);
  }

}
}
