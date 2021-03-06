#include <cassert>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "shape.h"
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
  void gemmCPU(float *alpha, float *beta, float *omega, float *gamma,
      size_t gamma_row, size_t dim_num, size_t gamma_col,
      bool t_alpha,  bool t_beta, float a, float b) {
    float value;
    for (size_t i = 0; i < gamma_row; ++i)
      for (size_t j = 0; j < gamma_col; ++j) {
        value = 0;
        for (size_t k = 0; k < dim_num; ++k) {
          value += alpha[t_alpha?pos(k, i, gamma_row):pos(i, k, dim_num)]
            * beta[t_beta?pos(j, k, dim_num):pos(k, j, gamma_col)];
        }
        gamma[pos(i, j, gamma_col)] = value * a + ((omega!=NULL)?(omega[j] * b):(0));
      }
  }


  /* Reduce matrix to vector
   *
   *  Specification:
   *   - m (row, col)
   *   - v (row) if reduced_dim is 2, else (col)
   */
  void reduceMatrixCPU(float* m, float* v, int row, int col, int reduced_dim) {
    assert(reduced_dim <= 2);
    float value = 0;
    if (reduced_dim == 1) {
      for (int i = 0; i < col; ++i) {
        value = 0;
        for (int j = 0; j < row; ++j)
          value += m[pos(j, i, col)];
        v[i] = value;
      }
    } else {
      for (int i = 0; i < row; ++i) {
        value = 0;
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
  void eleSubtractCPU(float* alpha, float* beta,
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
  void sigmoidOpCPU(float* alpha, float* beta, int row, int col) {
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
  void eleSquareCPU(float* alpha, float* beta, int row, int col) {
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
  void reduceToValueCPU(float* alpha, float* value, int row, int col) {
    float tmp = 0;
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        tmp += alpha[pos(i, j, col)];
    *value = tmp;
  }

  /* linearOp
   *  beta = alpha * scale + bias
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOpCPU(float* alpha, float* beta, int row,
      int col, float scale, float bias) {
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
  void multiEleCPU(float* alpha, float* beta, float* omega, int row, int col) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        omega[pos(i, j, col)] = alpha[pos(i, j, col)] * beta[pos(i, j, col)];
  }

  /* multiEleInplace
   *  beta = alpha * beta
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta  (row, col)
   */
  void multiEleInplaceCPU(float* alpha, float* beta, int row, int col) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] *= alpha[pos(i, j, col)];
  }

  /* linearOpInplace 
   *  beta = beta * a + alpha*b + c
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void linearOpInplaceCPU(float* alpha, float* beta, int row, int col, float a, float b, float c) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] = beta[pos(i, j, col)] * a +
          alpha[pos(i, j, col)] * b + c;
  }

  /* UpdateWeightWithReg
   *  w = w - delta*lr - reg*w
   *
   *  Specification:
   *   - alpha (row, col)
   *   - beta (row, col)
   */
  void UpdateWeightWithRegCPU(float* delta, float* weight, int row, int col, float lr, float reg) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        weight[pos(i, j, col)] = weight[pos(i, j, col)]*(1-reg) -
          delta[pos(i, j, col)] * lr;
  }

  /* getCorrectlyRecognized 
   *  get total number of correctly recognized value
   *
   *  Specification:
   *   - predicts (row, col)
   *   - targets (row, col)
   */
  int getCorrectlyRecognizedCPU(float* predicts, float* targets, int row, int col) {
    int matched_num = 0, max_p_idx, max_t_idx;
    float max_p_value, max_t_value;

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
  void softmaxCPU(float* alpha, float* beta, int row, int col) {
    for (int i = 0; i < row; ++i) {
      float sum = 0;
      for (int j = 0; j < col; ++j)
        sum += exp(alpha[pos(i, j, col)]);
      for (int j = 0; j < col; ++j)
        beta[pos(i, j, col)] = exp(alpha[pos(i, j, col)]) / sum;
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
  float negLogLikelihoodCPU(float* predicts, float* targets, int row, int col) {
    float loss = 0;
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        loss += targets[pos(i, j, col)] * log(predicts[pos(i, j, col)]);
    return -loss;
  }

  void normalizeCPU(float* data, int row, int col, float mean, float std) {
    for (int i = 0; i < row; ++i)
      for (int j = 0; j < col; ++j)
        data[pos(i, j, col)] = (data[pos(i, j, col)] - mean)/std;
  }

  std::tuple<float, float> getStdAndMeanCPU(float *data, shape sh) {
    float mean = 0;
    float std = 0;
    size_t total = sh.getTotal();
    for (size_t i = 0; i < total; ++i)
      mean += data[i];
    mean /= total;
    for (size_t i = 0; i < total; ++i)
      std += pow(mean - data[i], 2);
    std = sqrt(std/(total-1));
    return std::make_tuple(mean, std);
  }

}
}
