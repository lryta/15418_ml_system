#include "layer.h"

namespace MLLib {

class FullyConnectedLayer:Layer {
public:
  FullyConnectedLayer(size_t input_dim, size_t output_dim):
    weight_t(input_dim, output_dim), bias_(output_dim)  {
  }

private:
  tensor weight_, bias_;
};

}
