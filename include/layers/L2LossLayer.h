#include "tensor.h"
#include "layer.h"

namespace TinyML {

class L2LossLayer:LossLayer {
public:
  L2LossLayer(vecotr<shape> ins):LossLayer(vecotr<shape> ins), loss_(0), tensor_(ins[0].getShape()) {
  }

  vector<Tensor&> getParam() {
    return {};
  }

  void initIntermediateState(vector<shape> &ins, vector<shape> &ous) {
  }

private:
  float loss_;
  tensor inter_;
};

}
