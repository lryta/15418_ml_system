#ifndef _TINYML_LAYERS_NEGLOGLIKELILAYER_H
#define _TINYML_LAYERS_NEGLOGLIKELILAYER_H

#include "tensor.h"
#include "layer.h"

namespace TinyML {

class negLogLikelilayer: public Losslayer {
 public:
  negLogLikelilayer(vector<shape> &ins, vector<shape> &ous):Losslayer(ins, ous),
    correctlyRecognizedNum_(0), loss_(0)
  {}

  vector<tensor*> getParam() {
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

  virtual void forward(vector<tensor*> ins, vector<tensor*> ous);
  virtual void backward(vector<tensor*> ins, vector<tensor*> ous);

 private:
  int correctlyRecognizedNum_;
  float loss_;
  tensor predict_;
};

}

#endif
