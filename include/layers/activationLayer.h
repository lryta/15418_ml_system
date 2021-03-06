#ifndef _TINYML_LAYERS_ACTIVATIONLAYER_H
#define _TINYML_LAYERS_ACTIVATIONLAYER_H

#include "layer.h"
#include "shape.h"

namespace TinyML {

class Activationlayer: public layer {
 public:
  Activationlayer(vector<shape> &ins, vector<shape> &ous):
    layer(ins, ous) {}

  // For activation layer, the output should be exactly the same
  void inferShape(vector<shape> &ins, vector<shape> &ous) {
    assert(ins.size() == 1);
    ous.clear();
    for (auto const& in : ins)
      ous.push_back(in);
  }

  // Usually no weight
  void initWeight(vector<shape> &ins, vector<shape> &ous) {}

  // Usually no weight
  vector<tensor*> getParam() { return {}; }

  virtual void forward(vector<tensor*> ins, vector<tensor*> ous) = 0;
  virtual void backward(vector<tensor*> ins, vector<tensor*> ous) = 0;
};

class Sigmoidlayer: public Activationlayer {
 public:
  Sigmoidlayer(vector<shape> &ins, vector<shape> &ous):
    Activationlayer(ins, ous) { }

  void initIntermediateState(vector<shape> &ins, vector<shape> &ous)
  { inter_ = tensor(ins[0]); }

  virtual void forward(vector<tensor*> ins, vector<tensor*> ous);
  virtual void backward(vector<tensor*> ins, vector<tensor*> ous);
 private:
  tensor inter_;
};

}

#endif
