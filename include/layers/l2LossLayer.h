#ifndef _TINYML_LAYERS_L2LOSSLAYER_H
#define _TINYML_LAYERS_L2LOSSLAYER_H

#include "tensor.h"
#include "layer.h"

namespace TinyML {

class L2Losslayer:Losslayer {
 public:
  L2Losslayer(vecotr<shape> ins):Losslayer(vecotr<shape> ins),
    correctlyRecognizedNum_(0), loss_(0) 
  {}

  vector<tensor&> getParam() {
    return {};
  }

  void initIntermediateState(vector<shape> &ins, vector<shape> &ous) {
    inter_ = tensor(ins[0]);
  }

  float getLoss() {
    return loss_;
  }

  int correctlyRecognizedDataNum() {
    return correctlyRecognizedNum_;
  }

 private:
  int correctlyRecognizedNum_;
  float loss_;
  tensor inter_;
};

}

#endif
