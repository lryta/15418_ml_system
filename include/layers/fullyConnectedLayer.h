#ifndef _TINYML_LAYERS_FULLYCONNECTEDLAYER_H
#define _TINYML_LAYERS_FULLYCONNECTEDLAYER_H

#include "layer.h"

namespace TinyML {

class FullyConnectedlayer: public layer {
 public:
  FullyConnectedlayer(vector<shape> &ins, vector<shape> &ous, int inter_dim):
    layer(ins, ous),inter_dim_(inter_dim) {
  }

  virtual void inferShape(vector<shape> &ins, vector<shape> &ous);
  virtual void initWeight(vector<shape> &ins, vector<shape> &ous);
  virtual void initIntermediateState(vector<shape> &ins, vector<shape> &ous) {}
  virtual vector<tensor*> getParam();

  virtual void forward(vector<tensor*> ins, vector<tensor*> ous);
  virtual void backward(vector<tensor*> ins, vector<tensor*> ous);

private:
  size_t inter_dim_;
  tensor weight_, bias_;
};

}

#endif
