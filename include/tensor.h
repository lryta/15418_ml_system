#include <stdlib.h>

#include "shape.h"

namespace MLLib {

class Tensor {
public:
  // matrix M * N
  Tensor(shape s):shape_(s) {
    ptr_ = (float*) calloc(shape_.getTotal() * sizeof(float));
  }

  tuple<> getShape() {
  }

private:
  shape shape_;
  float *ptr_;
};

}
