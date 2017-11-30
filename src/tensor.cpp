#include <vector>
#include <cstring>
#include <cassert>
#include "tensor.h"

namespace TinyML {
using std::vector;

tensor::tensor():shape_(0), data_(NULL), grad_(NULL)
  {}

tensor::tensor(shape s):shape_(s), grad_(NULL) {
  data_ = (float*) calloc(shape_.getTotal(), sizeof(float));
}

tensor::tensor(vector<float> *v):shape_{v->size()} {
  data_ = (float*) malloc(shape_.getTotal() * sizeof(float));
  for (int i = 0; i < v->size(); ++i)
    data_[i] = v->at(i);
}

tensor::tensor(vector<vector<float>> *v):shape_{v->size(), v->at(0).size()} {
  data_ = (float*) malloc(shape_.getTotal() * sizeof(float));
  int cnt = -1;
  for (int i = 0; i < v->size(); ++i)
    for (int j = 0; j < v->at(0).size(); ++j)
      data_[++cnt] = v->at(i)[j];
}

tensor& tensor::operator=(const tensor& t) {
  if (grad_ != NULL) {
    delete grad_;
    grad_ = NULL;
  }
  if (data_ != NULL) {
    delete data_;
    data_ = NULL;
  }
  shape_ = t.shape_;
  if (t.data_ != NULL) {
    data_ = (float*) malloc(shape_.getTotal() * sizeof(float));
    memcpy(data_, t.data_, shape_.getTotal() * sizeof(float));
  }

  if (t.grad_ != NULL) {
    grad_ = (float*) malloc(shape_.getTotal() * sizeof(float));
    memcpy(grad_, t.grad_, shape_.getTotal() * sizeof(float));
  }
}

tensor::~tensor() {
  if (grad_ != NULL)
    delete grad_;
  if (data_ != NULL)
    delete data_;
}

shape tensor::getShape() {
  assert(shape_.getTotal() != 0);
  return shape_;
}

// Would be extended to getCPUPtr & getGPUPtr
float* tensor::getData() {
  assert(data_ != NULL);
  return data_;
}

float* tensor::getGrad() {
  if (grad_ == NULL)
    grad_ = (float*) calloc(shape_.getTotal(), sizeof(float));
  return grad_;
}

}
