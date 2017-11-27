#include "tensor.h"
#include "layer.h"

namespace TinyML {

class L2LossLayer:LossLayer {
 public:
  L2LossLayer(vecotr<shape> ins):LossLayer(vecotr<shape> ins),
    correctlyRecognizedNum_(0), loss_(0) 
  {}

  vector<Tensor&> getParam() {
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
