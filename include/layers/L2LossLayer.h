#include "tensor.h"
#include "layer.h"

namespace MLLib {

class L2LossLayer:LossLayer{
public:
  L2LossLayer(vecotr<shape> ins):LossLayer(vecotr<shape> ins), loss_(0), tensor_(ins[0].getShape()) {
  }

private:
  float loss_;
  tensor inter_;
};

}
