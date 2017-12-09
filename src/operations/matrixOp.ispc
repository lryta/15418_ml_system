#define pos(i, j, col_num) ((col_num) * (i) + (j))
#define min(x, y)  (((x)<(y))?(x):(y))
#define TASK_NUM 32

task void  gemmISPCTask(uniform float alpha[], uniform float beta[],
      uniform float omega[], uniform float gamma[],
      uniform size_t gamma_row, uniform size_t dim_num,
      uniform size_t gamma_col, uniform bool t_alpha,
      uniform bool t_beta, uniform float a,
      uniform float b, uniform size_t row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, gamma_row);

  // TODO: check to reverse the order
  foreach (i = ystart ... yend, j = 0 ... gamma_col) {
    float value = 0;
    for (size_t k = 0; k < dim_num; ++k) {
      value += alpha[t_alpha?pos(k, i, gamma_row):pos(i, k, dim_num)]
        * beta[t_beta?pos(j, k, dim_num):pos(k, j, gamma_col)];
    }
    gamma[pos(i, j, gamma_col)] = value * a + ((omega!=NULL)?(omega[j] * b):(0));
  }
}

export void gemmISPC(uniform float alpha[], uniform float beta[],
      uniform float omega[], uniform float gamma[],
      uniform size_t gamma_row, uniform size_t dim_num,
      uniform size_t gamma_col, uniform bool t_alpha,
      uniform bool t_beta, uniform float a,
      uniform float b)
{
  uniform int rowsPerTask = gamma_row / TASK_NUM;
  // create 32 tasks
  launch[TASK_NUM] gemmISPCTask(alpha, beta,
                             omega, gamma,
                             gamma_row, dim_num,
                             gamma_col, t_alpha,
                             t_beta, a,
                             b, rowsPerTask); 
}

task void reduceMatrixISPCTask(uniform float m[], uniform float v[],
    uniform int row, uniform int col,
    uniform int reduced_dim, uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task,
      (reduced_dim == 1)?(col):(row));

  float value = 0;
  if (reduced_dim == 1) {
    foreach (i = ystart ... yend) {
      value = 0;
      for (int j = 0; j < row; ++j)
        value += m[pos(j, i, col)];
      v[i] = value;
    }
  } else {
    foreach (i = ystart ... yend) {
      value = 0;
      for (int j = 0; j < col; ++j)
        value += m[pos(i, j, col)];
      v[i] = value;
    }
  }
}

export void reduceMatrixISPC(uniform float m[], uniform float v[],
    uniform int row, uniform int col,
    uniform int reduced_dim) {
  uniform int rowsPerTask = (reduced_dim == 1)?(col/TASK_NUM):(row/TASK_NUM);

  launch[TASK_NUM] reduceMatrixISPCTask(m, v,
      row, col, reduced_dim, rowsPerTask); 
}

task void eleSubtractISPCTask(uniform float alpha[], uniform float beta[],
    uniform float omega[], uniform int row, uniform int col,
    uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    omega[pos(i, j, col)] = alpha[pos(i, j, col)] - beta[pos(i, j, col)];
  }
}

export void eleSubtractISPC(uniform float alpha[], uniform float beta[],
    uniform float omega[], uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] eleSubtractISPCTask(alpha, beta,
      omega, row, col, rowsPerTask); 
}

task void sigmoidOpISPCTask(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col, uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    beta[pos(i, j, col)] = 1 / (1 + exp(-alpha[pos(i, j, col)]));
  }
}

export void sigmoidOpISPC(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] sigmoidOpISPCTask(alpha, beta,
      row, col, rowsPerTask); 
}

task void eleSquareISPCTask(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col, uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    beta[pos(i, j, col)] = pow(alpha[pos(i, j, col)], 2);
  }
}

export void eleSquareISPC(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] eleSquareISPCTask(alpha, beta,
      row, col, rowsPerTask); 
}

task void reduceToValueISPCTask(uniform float alpha[], uniform float task_res[],
    uniform int row, uniform int col, uniform int row_per_task) {
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);
  float sum = 0;

  foreach (j = 0 ...col, i = ystart ... yend) {
    sum += alpha[pos(i, j, col)];
  }

  task_res[taskIndex] = reduce_add(sum);
}

export void reduceToValueISPC(uniform float alpha[], uniform float value[],
    uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;
  uniform float task_res[TASK_NUM];
  uniform float task_num = TASK_NUM;

  launch[TASK_NUM] reduceToValueISPCTask(alpha, task_res,
      row, col, rowsPerTask); 
  sync;

  float total_value = 0;
  foreach (i = 0...task_num)
    total_value += task_res[i];
  value[0] = reduce_add(total_value);
}

task void linearOpISPCTask(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col, uniform float scale,
    uniform float bias, uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    beta[pos(i, j, col)] = alpha[pos(i, j, col)] * scale + bias;
  }
}

export void linearOpISPC(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col, uniform float scale,
    uniform float bias) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] linearOpISPCTask(alpha, beta,
      row, col, scale, bias, rowsPerTask);
}

task void multiEleISPCTask(uniform float alpha[], uniform float beta[],
    uniform float omega[], uniform int row, uniform int col,
    uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    omega[pos(i, j, col)] = alpha[pos(i, j, col)] * beta[pos(i, j, col)];
  }
}

export void multiEleISPC(uniform float alpha[], uniform float beta[],
    uniform float omega[], uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] multiEleISPCTask(alpha, beta,
      omega, row, col, rowsPerTask);
}

task void multiEleInplaceISPCTask(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col,
    uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    beta[pos(i, j, col)] *= alpha[pos(i, j, col)];
  }
}

export void multiEleInplaceISPC(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] multiEleInplaceISPCTask(alpha, beta,
      row, col, rowsPerTask);
}

task void negLogLikelihoodISPCTask(uniform float predicts[], uniform float targets[],
    uniform int row, uniform int col,
    uniform int row_per_task, uniform float task_loss[]) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);
  float loss = 0;

  foreach (j = 0 ... col, i = ystart ... yend) {
    loss += targets[pos(i, j, col)] * log(predicts[pos(i, j, col)]);
  }

  task_loss[taskIndex] = -reduce_add(loss);
}

export uniform float negLogLikelihoodISPC(uniform float predicts[], uniform float targets[],
    uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;
  uniform float task_loss[TASK_NUM];
  uniform float total_loss;
  uniform float task_num = TASK_NUM;

  launch[TASK_NUM]negLogLikelihoodISPCTask(predicts, targets,
      row, col, rowsPerTask, task_loss);
  sync;

  float sum_loss = 0;
  foreach (i = 0...task_num)
    sum_loss += task_loss[i];
  total_loss = reduce_add(sum_loss);

  return total_loss;
}

task void softmaxISPCTask(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col,
    uniform int row_per_task) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  float sum;
  foreach (i = ystart ... yend) {
    sum = 0;
    for (int j = 0; j < col; ++j)
      sum += exp(alpha[pos(i, j, col)]);
    for (int j = 0; j < col; ++j)
      beta[pos(i, j, col)] = exp(alpha[pos(i, j, col)]) / sum;
  }

}

export void softmaxISPC(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] softmaxISPCTask(alpha, beta,
      row, col, rowsPerTask);
}

task void getCorrectlyRecognizedISPCTask(uniform float predicts[],
    uniform float targets[], uniform int row, uniform int col,
    uniform int row_per_task, uniform int task_res[]) {
  // taskIndex is an ISPC built-in
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);
  
  float max_p_value, max_t_value;
  int max_p_idx, max_t_idx, matched_num = 0; 
  foreach (i = ystart ... yend) {
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

  task_res[taskIndex] = reduce_add(matched_num);
}

export uniform int getCorrectlyRecognizedISPC(uniform float predicts[],
    uniform float targets[], uniform int row, uniform int col) {
  uniform int rowsPerTask = row/TASK_NUM;
  uniform int total_matched_num;
  uniform int task_res[TASK_NUM];
  uniform float task_num = TASK_NUM;

  launch[TASK_NUM] getCorrectlyRecognizedISPCTask(predicts, targets,
      row, col, rowsPerTask, task_res);
  sync;

  int matched_num = 0;
  foreach (i = 0...task_num)
    matched_num += task_res[i];
  total_matched_num = reduce_add(matched_num);

  return total_matched_num;
}

task void linearOpInplaceISPCTask(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col, uniform float a,
    uniform float b, uniform float c, uniform int row_per_task) {
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    beta[pos(i, j, col)] = beta[pos(i, j, col)] * a + alpha[pos(i, j, col)] * b + c;
  }
}

export void linearOpInplaceISPC(uniform float alpha[], uniform float beta[],
    uniform int row, uniform int col, uniform float a,
    uniform float b, uniform float c) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] linearOpInplaceISPCTask(alpha, beta,
      row, col, a, b, c, rowsPerTask);
}

task void UpdateWeightWithRegISPCTask(uniform float delta[], uniform float weight[],
    uniform int row, uniform int col, uniform float lr,
    uniform float reg, uniform int row_per_task) {
  uniform int ystart = taskIndex * row_per_task;
  uniform int yend = min(ystart + row_per_task, row);

  foreach (j = 0 ... col, i = ystart ... yend) {
    weight[pos(i, j, col)] = weight[pos(i, j, col)]*(1-reg) -
      delta[pos(i, j, col)] * lr;
  }
}

export void UpdateWeightWithRegISPC(uniform float delta[], uniform float weight[],
    uniform int row, uniform int col, uniform float lr,
    uniform float reg) {
  uniform int rowsPerTask = row/TASK_NUM;

  launch[TASK_NUM] UpdateWeightWithRegISPCTask(delta, weight,
      row, col, lr, reg, rowsPerTask);
}
