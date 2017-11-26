#include "layer.h"

namespace TinyML {

class FullyConnectedLayer:Layer {
 public:
  FullyConnectedLayer(vector<shape> ins, vector<shape> ous, int inter_dim):
    Layer(ins, ous),inter_dim_(inter_dim) {
  }

private:
  size_t inter_dim_;
  tensor weight_, bias_;
};

}
