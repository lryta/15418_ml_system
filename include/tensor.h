#include <stdlib.h>

#include "shape.h"

namespace MLLib {

class Tensor {
 public:
  // matrix M * N
  Tensor(shape s):shape_(s) {
    ptr_ = (float*) calloc(shape_.getTotal() * sizeof(float));
  }

  // TODO: copy constructor of shape
  Tensor(Tensor& src) {
    shape_ = src.shape_;
    ptr_ = (float*) calloc(shape_.getTotal() * sizeof(float));
  }

  // TODO: Add Grad()
  // TODO: Add De-Allocator()

  shape getShape() {
    return shape_;
  }

  // Would be extended to getCPUPtr & getGPUPtr
  float* getPtr() {
    return ptr_;
  }

 private:
  shape shape_;
  float *ptr_;
};

}
