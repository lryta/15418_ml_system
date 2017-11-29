#ifndef _TINYML_LAYERS_L2LOSSLAYER_H
#define _TINYML_LAYERS_L2LOSSLAYER_H

#include <vector>

#include "tensor.h"
#include "layer.h"
#include "shape.h"

namespace TinyML {

using std::vector;

class L2Losslayer: public Losslayer {
 public:
  L2Losslayer(vector<shape> &ins, vector<shape> &ous):Losslayer(ins, ous),
    correctlyRecognizedNum_(0), loss_(0) {}

  vector<tensor*> getParam() {
    return {};
  }

  void initIntermediateState(vector<shape> &ins, vector<shape> &ous) {
    inter_ = tensor(ins[0]);
    inter_square_ = tensor(ins[0]);
  }

  float getLoss() {
    return loss_;
  }

  int correctlyRecognizedDataNum() {
    return correctlyRecognizedNum_;
  }

  virtual void forward(vector<tensor*> ins, vector<tensor*> ous);
  virtual void backward(vector<tensor*> ins, vector<tensor*> ous);

 private:
  int correctlyRecognizedNum_;
  float loss_;
  tensor inter_, inter_square_;
};

}

#endif
