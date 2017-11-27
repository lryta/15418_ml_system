#include <stdlib.h>

#include "shape.h"

namespace TinyML {

class tensor {
 public:
  tensor();
  tensor(shape s);
  tensor(vector<float> *v);
  tensor(vector<vector<float>> *v);
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
