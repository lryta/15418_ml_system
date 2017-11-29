#ifndef _TINYML_TENSOR_H
#define _TINYML_TENSOR_H

#include <stdlib.h>
#include <vector>

#include "shape.h"

namespace TinyML {

class tensor {
 public:
  tensor();
  tensor(shape s);
  tensor(std::vector<float> *v);
  tensor(std::vector<std::vector<float>> *v);
  tensor(const tensor&) = delete;

  ~tensor();

  shape getShape();
  // TODO: extended to support GPU pointer
  // e.g. getCPUData. getGPUGrad
  float* getData();
  float* getGrad();

 private:
  shape shape_;
  float *data_, *grad_;
};

}


#endif
