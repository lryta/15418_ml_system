#ifndef _TINYML_LAYERS_FULLYCONNECTEDLAYER_H
#define _TINYML_LAYERS_FULLYCONNECTEDLAYER_H

#include "layer.h"

namespace TinyML {

class FullyConnectedlayer:layer {
 public:
  FullyConnectedlayer(vector<shape> ins, vector<shape> ous, int inter_dim):
    layer(ins, ous),inter_dim_(inter_dim) {
  }

private:
  size_t inter_dim_;
  tensor weight_, bias_;
};

}

#endif
