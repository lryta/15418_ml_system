#include "tensor.h"

namespace MLLib {

tensor::tensor(shape s):shape_(s), grad_(NULL) {
  data_ = (float*) calloc(shape_.getTotal() * sizeof(float));
}

tensor::tensor(vector<float> *v):shape_{v.size()} {
  data_ = (float*) malloc(shape_.getTotal() * sizeof(float));
  for (int i = 0; i < v.size(); ++i)
    data_[i] = v->at(i);
}

tensor::tensor(vector<vector<float>> *v):shape_{v.size(), v[0].size()} {
  data_ = (float*) malloc(shape_.getTotal() * sizeof(float));
  int cnt = -1;
  for (int i = 0; i < v.size(); ++i)
    for (int j = 0; j < v[0].size(); ++j)
      data_[++cnt] = v->at(i)[j];
}

~tensor::tensor() {
  if (grad_ != NULL)
    delete grad_;
  delete data_;
}

shape tensor::getShape() {
  return shape_;
}

// Would be extended to getCPUPtr & getGPUPtr
float* tensor::getData() {
  return data_;
}

float* tensor::getGrad() {
  if (grad_ == NULL)
    grad_ = (float*) calloc(shape_.getTotal() * sizeof(float));
  return grad_;
}

}
