#ifndef _TINYML_LAYERS_NEGLOGLIKELILAYER_H
#define _TINYML_LAYERS_NEGLOGLIKELILAYER_H

#include "tensor.h"
#include "layer.h"

namespace TinyML {

class negLogLikelilayer:Losslayer {
 public:
  negLogLikelilayer(vecotr<shape> ins):Losslayer(vecotr<shape> ins),
    correctlyRecognizedNum_(0), loss_(0)
  {}

  vector<tensor&> getParam() {
    return {};
  }

  void initIntermediateState(vector<shape> &ins, vector<shape> &ous) {
    predict_ = tensor(ins[0]);
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
  tensor predict_;
};

}

#endif
